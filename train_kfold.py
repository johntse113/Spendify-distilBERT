# python train_kfold.py --data_dir ./data/all --output_dir ./saved_model_kfold --k_folds 5 --epochs 20 --batch_size 8

import os
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report as ner_classification_report

from receipt_dataset import (
    NER_LABEL2ID, NER_ID2LABEL,
    CATEGORY_LABEL2ID, CATEGORY_ID2LABEL,
    MAX_SEQ_LEN, MODEL_NAME,
    DEFAULT_CATEGORY,
    ocr_image, word_level_bio_labels,
)
from nlp_model import build_model, NUM_NER_LABELS, NUM_CAT_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CombinedReceiptDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        tokenizer: DistilBertTokenizerFast,
        max_len: int = MAX_SEQ_LEN,
        ocr_cache_dir: str = None,
    ):
        from receipt_dataset import LABEL_FIELD_MAP, fuzzy_find_span, preprocess_image

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Dict] = []

        data_path = Path(data_dir)
        jpg_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.JPG"))

        cache_dir = Path(ocr_cache_dir) if ocr_cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for jpg_path in sorted(jpg_files):  # sorted for reproducibility
            txt_path = jpg_path.with_suffix(".txt")
            if not txt_path.exists():
                log.warning(f"Skipping {jpg_path.name} – no label file")
                continue

            if cache_dir:
                cache_file = cache_dir / (jpg_path.stem + ".ocr.txt")
                if cache_file.exists():
                    ocr_text = cache_file.read_text(encoding="utf-8")
                else:
                    ocr_text = ocr_image(str(jpg_path))
                    cache_file.write_text(ocr_text, encoding="utf-8")
            else:
                ocr_text = ocr_image(str(jpg_path))

            if not ocr_text.strip():
                log.warning(f"Empty OCR for {jpg_path.name}; skipping.")
                continue

        
            try:
                label_json = json.loads(txt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning(f"Invalid JSON in {txt_path.name}; skipping.")
                continue

            cat_raw = label_json.get("category", DEFAULT_CATEGORY)
            cat_id = CATEGORY_LABEL2ID.get(cat_raw, CATEGORY_LABEL2ID[DEFAULT_CATEGORY])

            self.samples.append({
                "ocr_text": ocr_text,
                "label_json": label_json,
                "cat_id": cat_id,
            })

        log.info(f"CombinedReceiptDataset: {len(self.samples)} samples from '{data_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        ocr_text = s["ocr_text"]
        label_json = s["label_json"]
        cat_id = s["cat_id"]

        words, word_tags = word_level_bio_labels(ocr_text, label_json)

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)

        ner_aligned = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                ner_aligned.append(-100)          # CLS / SEP / PAD → ignore
            elif wid != prev_wid:
                ner_aligned.append(NER_LABEL2ID.get(word_tags[wid], 0))
            else:
                ner_aligned.append(-100)          # continuation sub-token → ignore
            prev_wid = wid

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ner_labels":     torch.tensor(ner_aligned, dtype=torch.long),
            "cat_labels":     torch.tensor(cat_id,      dtype=torch.long),
        }



def decode_ner_preds(preds: np.ndarray, labels: np.ndarray):
    """Convert flat arrays to seqeval-compatible lists-of-lists, ignoring -100."""
    pred_seqs, true_seqs = [], []
    for pred_row, label_row in zip(preds, labels):
        pred_seq, true_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_seq.append(NER_ID2LABEL.get(p, "O"))
            true_seq.append(NER_ID2LABEL.get(l, "O"))
        pred_seqs.append(pred_seq)
        true_seqs.append(true_seq)
    return pred_seqs, true_seqs



def run_fold(
    fold: int,
    total_folds: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args,
    tokenizer: DistilBertTokenizerFast,
    output_dir: Path,
    global_best_val_loss: float,
) -> float:

    model = build_model(MODEL_NAME).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_fold_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_labels     = batch["ner_labels"].to(device)
            cat_labels     = batch["cat_labels"].to(device)

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ner_labels=ner_labels,
                cat_labels=cat_labels,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        all_ner_preds,  all_ner_labels  = [], []
        all_cat_preds,  all_cat_labels  = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                ner_labels     = batch["ner_labels"].to(device)
                cat_labels     = batch["cat_labels"].to(device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    cat_labels=cat_labels,
                )
                total_val_loss += out["loss"].item()

                ner_preds = out["ner_logits"].argmax(dim=-1).cpu().numpy()
                cat_preds = out["cat_logits"].argmax(dim=-1).cpu().numpy()

                all_ner_preds.extend(ner_preds)
                all_ner_labels.extend(ner_labels.cpu().numpy())
                all_cat_preds.extend(cat_preds)
                all_cat_labels.extend(cat_labels.cpu().numpy())

        avg_val  = total_val_loss / len(val_loader)
        cat_acc  = accuracy_score(all_cat_labels, all_cat_preds)

        log.info(
            f"Fold {fold}/{total_folds} | Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f} | "
            f"cat_acc={cat_acc:.3f}"
        )

        try:
            pred_seqs, true_seqs = decode_ner_preds(
                np.array(all_ner_preds), np.array(all_ner_labels)
            )
            ner_report = ner_classification_report(true_seqs, pred_seqs, zero_division=0)
            log.info(f"\nNER report (fold {fold}, epoch {epoch}):\n{ner_report}")
        except Exception as e:
            log.warning(f"seqeval report failed: {e}")

        # Save if best across all folds
        if avg_val < global_best_val_loss:
            global_best_val_loss = avg_val
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            log.info(
                f" ✓ New global best saved "
                f"(fold {fold}, epoch {epoch}, val_loss={avg_val:.4f})"
            )

        if avg_val < best_fold_val_loss:
            best_fold_val_loss = avg_val

    return best_fold_val_loss, global_best_val_loss


# K-Fold
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    full_ds = CombinedReceiptDataset(
        args.data_dir,
        tokenizer,
        max_len=args.max_len,
        ocr_cache_dir=os.path.join(args.output_dir, "ocr_cache"),
    )

    if len(full_ds) < args.k_folds:
        raise ValueError(
            f"Dataset has only {len(full_ds)} samples but k_folds={args.k_folds}. "
            f"Reduce --k_folds or add more data."
        )

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    indices = list(range(len(full_ds)))

    fold_best_losses: List[float] = []
    global_best_val_loss = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        log.info(
            f"\n{'='*60}\n"
            f"  FOLD {fold}/{args.k_folds}  "
            f"(train={len(train_idx)} samples, val={len(val_idx)} samples)\n"
            f"{'='*60}"
        )

        train_sub = Subset(full_ds, train_idx)
        val_sub   = Subset(full_ds, val_idx)

        train_loader = DataLoader(
            train_sub, batch_size=args.batch_size, shuffle=True,  num_workers=2
        )
        val_loader = DataLoader(
            val_sub,   batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        best_fold_loss, global_best_val_loss = run_fold(
            fold=fold,
            total_folds=args.k_folds,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
            tokenizer=tokenizer,
            output_dir=output_dir,
            global_best_val_loss=global_best_val_loss,
        )
        fold_best_losses.append(best_fold_loss)
        log.info(f"Fold {fold} best val loss: {best_fold_loss:.4f}")

    mean_loss = sum(fold_best_losses) / len(fold_best_losses)
    log.info("\n" + "="*60)
    log.info(f"K-Fold training complete ({args.k_folds} folds).")
    log.info(f"Per-fold best val losses: {[round(l,4) for l in fold_best_losses]}")
    log.info(f"Mean val loss across folds: {mean_loss:.4f}")
    log.info(f"Global best val loss:       {global_best_val_loss:.4f}")
    log.info(f"Best model saved to:        {output_dir}")



def parse_args():
    p = argparse.ArgumentParser(description="Train MultiTask Receipt NLP Model (K-Fold)")
    p.add_argument("--data_dir",    required=True,
                   help="Path to directory containing ALL receipt jpg+txt pairs")
    p.add_argument("--output_dir",  default="./saved_model",
                   help="Where to save the best checkpoint (lowest val loss across all folds)")
    p.add_argument("--k_folds",     type=int,   default=5,
                   help="Number of folds for cross-validation (default: 5)")
    p.add_argument("--epochs",      type=int,   default=20,
                   help="Training epochs per fold")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--max_len",     type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--warmup_ratio",type=float, default=0.1,
                   help="Fraction of total steps used for linear warm-up")
    p.add_argument("--max_grad_norm",type=float,default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
