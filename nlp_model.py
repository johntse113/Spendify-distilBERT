import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel, DistilBertConfig

NUM_NER_LABELS   = 7   # O B-MERCHANT I-MERCHANT B-DATE I-DATE B-TOTAL I-TOTAL
NUM_CAT_LABELS   = 5   # Food&Dining Transport Shopping Entertainment Utilities
NER_LOSS_WEIGHT  = 0.6
CAT_LOSS_WEIGHT  = 0.4


class MultiTaskReceiptModel(DistilBertPreTrainedModel):
    # Initialize with DistilBERT backbone 
    # Two heads for NER and Category Classification

    def __init__(self, config: DistilBertConfig):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.dropout    = nn.Dropout(config.dropout)
        self.ner_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, NUM_NER_LABELS),
        )

        self.category_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 4, NUM_CAT_LABELS),
        )

        self.post_init()

    # Forward pass returns dict with losses and logits for both tasks
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        ner_labels:     torch.Tensor = None,
        cat_labels:     torch.Tensor = None,
    ):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        cls_output      = sequence_output[:, 0, :]
        ner_logits = self.ner_head(sequence_output)
        cat_logits = self.category_head(cls_output)
        total_loss = None
        ner_loss   = None
        cat_loss   = None

        if ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = loss_fct(
                ner_logits.view(-1, NUM_NER_LABELS),
                ner_labels.view(-1),
            )

        if cat_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            cat_loss = loss_fct(cat_logits, cat_labels)

        if ner_loss is not None and cat_loss is not None:
            total_loss = NER_LOSS_WEIGHT * ner_loss + CAT_LOSS_WEIGHT * cat_loss
        elif ner_loss is not None:
            total_loss = ner_loss
        elif cat_loss is not None:
            total_loss = cat_loss

        return {
            "loss":       total_loss,
            "ner_loss":   ner_loss,
            "cat_loss":   cat_loss,
            "ner_logits": ner_logits,
            "cat_logits": cat_logits,
        }


def build_model(pretrained_name: str = "distilbert-base-uncased") -> MultiTaskReceiptModel:
    # Load DistilBERT config and initialize our multi-task model
    config = DistilBertConfig.from_pretrained(pretrained_name)
    model  = MultiTaskReceiptModel(config)
    base_state = DistilBertModel.from_pretrained(pretrained_name).state_dict()
    missing, unexpected = model.distilbert.load_state_dict(base_state, strict=False)
    if missing:
        print(f"[build_model] Missing keys (expected for heads): {len(missing)}")
    return model
