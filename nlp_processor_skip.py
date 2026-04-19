from html import entities
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import DistilBertTokenizerFast
from datetime import datetime
from receipt_dataset import (
    NER_ID2LABEL, CATEGORY_ID2LABEL, CATEGORY_LABEL2ID,
    DEFAULT_CATEGORY, MAX_SEQ_LEN, MODEL_NAME,
    ocr_image,
)
from nlp_model import MultiTaskReceiptModel, NUM_NER_LABELS, NUM_CAT_LABELS

log = logging.getLogger(__name__)


_MONTH_NAMES = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10,
    'november': 11, 'december': 12,
}

def _normalise_to_iso(
    day: int, month: int, year: int,
    hour: int = 0, minute: int = 0, second: int = 0,
    ampm: str = ""
) -> str:
    if year < 100:
        year += 2000

    ampm = ampm.lower().replace('.', '').strip()
    if ampm == 'pm' and hour != 12:
        hour += 12
    elif ampm == 'am' and hour == 12:
        hour = 0
    elif ampm in ('nn', 'noon'):
        hour, minute, second = 12, 0, 0
    elif ampm == 'midnight':
        hour, minute, second = 0, 0, 0

    try:
        dt = datetime(year, month, day, hour, minute, second)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:        # Invalid date
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

_DATE_PATTERNS = re.compile(
    r"""
    (?:
        \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}     # dd-mm-yyyy  dd/mm/yy
      | \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}        # yyyy-mm-dd
      | \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun
                    |Jul|Aug|Sep|Oct|Nov|Dec
                    |January|February|March|April
                    |May|June|July|August|September
                    |October|November|December)\s+\d{2,4}
      | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec
          |January|February|March|April|May|June|July|August
          |September|October|November|December)\s+\d{1,2}[,\s]+\d{4}
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


_DATE_FALLBACK_PATTERN = re.compile(
    r"""
    (?:
        (\d{1,2})
        [\s]*[./,\-][\s]*
        (\d{1,2})
        [\s]*[./,\-/][\s]*
        (\d{2,4})
    )

    (?:
        [\s]*[\s\-–—/|,]*[\s]*
        (\d{1,2})
        [\s]*:[\s]*
        (\d{2})
        (?:[\s]*:[\s]*(\d{2}))?
        [\s]*  
        (
            a\.m\.|p\.m\.|
            am|pm|
            nn|noon|midnight
        )?
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)



_CURRENCY_PREFIX = re.compile(
    r"^\$|^HKD|^RMB|^RM|^MYR|^USD|^SGD|^£|^€|^\u00a3|\u20b9",
    re.IGNORECASE,
)

# Category lexicon: list of (keyword, weight) pairs for each category.
CATEGORY_LEXICON: Dict[str, List[Tuple[str, float]]] = {
    "Food & Dining": [
        ("restaurant", 1.0), ("cafe", 1.0), ("coffee shop", 1.0), ("diner", 1.0),
        ("bistro", 1.0), ("brasserie", 1.0), ("eatery", 1.0), ("canteen", 0.9),
        ("food court", 1.0), ("food hall", 1.0), ("cafeteria", 0.9),
        ("takeaway", 0.8), ("takeout", 0.8), ("delivery", 0.6),
        ("catering", 0.9), ("buffet", 1.0), ("bakery", 0.9), ("patisserie", 0.9),
        ("deli", 0.9), ("delicatessen", 0.9), ("bar", 0.6), ("pub", 0.7),

        ("food", 0.8), ("meal", 1.0), ("breakfast", 1.0), ("lunch", 1.0),
        ("dinner", 1.0), ("brunch", 1.0), ("snack", 0.8), ("dessert", 0.8),
        ("coffee", 0.9), ("tea", 0.7), ("juice", 0.7), ("smoothie", 0.8),
        ("pizza", 1.0), ("burger", 1.0), ("sandwich", 0.9), ("salad", 0.8),
        ("sushi", 1.0), ("noodles", 0.9), ("pasta", 0.9), ("steak", 0.9),
        ("menu", 0.8), ("order", 0.5), ("cuisine", 0.9), ("grill", 0.8),
    ],
    "Transport": [
        ("taxi", 1.0), ("cab", 1.0), ("ride", 0.7), ("rideshare", 1.0),
        ("chauffeur", 0.9), ("shuttle", 0.9), ("transfer", 0.7),
        ("bus", 0.9), ("coach", 0.8), ("train", 0.9), ("subway", 1.0),
        ("metro", 1.0), ("tram", 1.0), ("rail", 0.9), ("transit", 0.8),
        ("flight", 1.0), ("airline", 1.0), ("airport", 0.8), ("ferry", 1.0),
        ("cruise", 0.9), ("ticket", 0.6), 
        ("postal", 0.9), ("courier", 0.9), ("delivery", 0.9),

        ("fuel", 1.0), ("petrol", 1.0), ("gasoline", 1.0), ("diesel", 1.0),
        ("gas station", 1.0), ("service station", 0.9), ("filling station", 0.9),
        ("toll", 1.0), ("highway", 0.8), ("motorway", 0.8), ("freeway", 0.8),
        ("parking", 1.0), ("car park", 1.0), ("valet", 0.9),
        ("car rental", 1.0), ("vehicle hire", 1.0), ("auto", 0.5),
        ("motor", 0.7), ("garage", 0.8), ("workshop", 0.6), ("service centre", 0.8),
        ("oil change", 1.0), ("tyre", 0.9), ("tire", 0.9),
    ],
    "Shopping": [
        ("mall", 0.9), ("shopping centre", 1.0), ("shopping center", 1.0),
        ("outlet", 0.8), ("shop", 0.7), ("boutique", 0.9), ("retail", 0.8),

        ("clothing", 0.9), ("apparel", 0.9), ("fashion", 0.8), ("footwear", 0.9),
        ("shoes", 0.8), ("accessories", 0.8), ("jewellery", 0.9), ("jewelry", 0.9),
        ("electronics", 0.9), ("appliance", 0.9), ("hardware", 0.8),
        ("stationery", 0.9), ("book", 0.7), ("bookstore", 1.0),
        ("grocery", 1.0), ("cosmetics", 0.9), ("pharmacy", 0.8), ("health & beauty", 0.9),
    ],
    "Entertainment": [
        ("cinema", 1.0), ("movie", 1.0), ("film", 0.9), ("theatre", 1.0),
        ("theater", 1.0), ("concert", 1.0), ("show", 0.7), ("performance", 0.8),
        ("festival", 0.8), ("event", 0.7), ("exhibition", 0.8), ("expo", 0.8),
        ("museum", 0.9), ("gallery", 0.8), ("theme park", 1.0), ("amusement", 0.9),
        ("zoo", 1.0), ("aquarium", 1.0),

        ("arcade", 1.0), ("gaming", 0.9), ("game", 0.8), ("bowling", 1.0),
        ("pool", 0.6), ("karaoke", 1.0), ("escape room", 1.0), ("gym", 0.8), 
        ("fitness", 0.8), ("yoga", 0.9), ("pilates", 0.9),
        ("spa", 0.9), ("massage", 0.9), ("salon", 0.8),

        ("streaming", 0.9), ("subscription", 0.6), ("download", 0.6),
        ("ticket", 0.6), ("admission", 0.9), ("entry fee", 0.9),
    ],
    "Utilities": [
        ("electricity", 1.0), ("electric", 1.0), ("power", 0.7), ("energy", 0.7),
        ("gas", 0.8), ("natural gas", 1.0), ("water", 0.8), ("sewage", 1.0),
        ("sewerage", 1.0), ("waste", 0.7), ("rubbish", 0.7), ("refuse", 0.7),

        ("internet", 1.0), ("broadband", 1.0), ("fibre", 1.0), ("fiber", 1.0),
        ("wifi", 0.9), ("wi-fi", 0.9), ("telephone", 0.9), ("phone", 0.7),
        ("mobile", 0.7), ("cellular", 0.8), ("postpaid", 1.0), ("prepaid", 0.8),
        ("data plan", 1.0), ("sim", 0.8), ("telecom", 0.9),

        ("bill", 0.8), ("invoice", 0.5), ("utility", 1.0), ("utilities", 1.0),
        ("account", 0.4), ("meter", 0.9), ("reading", 0.5), ("usage", 0.6),
        ("tariff", 0.8), ("rate", 0.4), ("supply", 0.5),
    ],
}


def lexicon_score(text: str) -> np.ndarray:
    scores = np.zeros(NUM_CAT_LABELS, dtype=np.float32)
    cat_order = ["Food & Dining", "Transport", "Shopping", "Entertainment", "Utilities"]
    
    text_lower = text.lower()
    for i, cat in enumerate(cat_order):
        for keyword, weight in CATEGORY_LEXICON[cat]:
            count = text_lower.count(keyword.lower())
            scores[i] += count * weight
    
    total = scores.sum()
    return scores / total if total > 0 else np.ones(NUM_CAT_LABELS) / NUM_CAT_LABELS


# Entity extraction from BIO tags
def _bio_to_entities(tokens: List[str], tags: List[str]) -> Dict[str, str]:
    entities: Dict[str, List[str]] = {}
    current_type = None
    current_tokens: List[str] = []

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_type:
                key = current_type.lower()
                if key not in entities:
                    entities[key] = " ".join(current_tokens)
            current_type   = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)
        else:
            if current_type:
                key = current_type.lower()
                if key not in entities:
                    entities[key] = " ".join(current_tokens)
            current_type   = None
            current_tokens = []

    if current_type:
        key = current_type.lower()
        if key not in entities:
            entities[key] = " ".join(current_tokens)

    return entities

def _is_plausible_date(day: int, month: int, year: int) -> bool:
    """Sanity-check extracted numeric components."""
    if year < 100:
        year += 2000  # two-digit year
    return (
        1 <= month <= 12
        and 1 <= day <= 31
        and 1900 <= year <= 2100
    )

def _extract_date_heuristic(raw_text: str) -> str:
    def _try_patterns(text: str) -> str:
        for m in _DATE_FALLBACK_PATTERN.finditer(text):
            try:
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            except (TypeError, ValueError):
                continue
            if not _is_plausible_date(d, mo, y):
                try:
                    if _is_plausible_date(int(m.group(3)), int(m.group(2)), int(m.group(1))):
                        d, mo, y = int(m.group(3)), int(m.group(2)), int(m.group(1))
                    else:
                        continue
                except (TypeError, ValueError):
                    continue

            hour   = int(m.group(4)) if m.group(4) else 0
            minute = int(m.group(5)) if m.group(5) else 0
            second = int(m.group(6)) if m.group(6) else 0
            ampm   = m.group(7) if m.group(7) else ""
            return _normalise_to_iso(d, mo, y, hour, minute, second, ampm)

        m = _DATE_PATTERNS.search(text)
        if m:
            raw = m.group(0).strip()
            for fmt in ('%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y',
                        '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d'):
                try:
                    dt = datetime.strptime(raw, fmt)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')  # time => 00:00:00
                except ValueError:
                    continue
            return raw
        return ""

    result = _try_patterns(raw_text)
    if result:
        return result
    collapsed = re.sub(r'\s+', '', raw_text)
    return _try_patterns(collapsed)


def _extract_items_heuristic(ocr_text: str) -> List[str]:
    items = []
    price_re = re.compile(r"(\d{1,6}[.,]\d{2})\s*$")
    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if price_re.search(line):
            item = price_re.sub("", line).strip()
            item = re.sub(r"\s{2,}", " ", item)
            if 3 <= len(item) <= 80:
                items.append(item)
    return items


def _extract_description_heuristic(ocr_text: str, skip_merchant: str = "") -> str:
    for line in ocr_text.splitlines():
        line = line.strip()
        if len(line) > 5 and line != skip_merchant:
            return line[:120]
    return ""


def _extract_merchant_heuristic(ocr_text: str) -> str:
    noise = re.compile(r'^[\s|\\/_\-=.,:;!?\[\](){}]+$')
    price_like = re.compile(r'\d{1,6}[.,]\d{2}')
    for line in ocr_text.splitlines():
        line = line.strip()
        if len(line) < 4:
            continue
        if noise.match(line):
            continue
        if price_like.search(line):
            continue
        alpha_ratio = sum(c.isalpha() for c in line) / len(line)
        if alpha_ratio > 0.5:
            return line[:80]
    return ""


def _extract_total_heuristic(ocr_text: str) -> str:
    total_keywords = re.compile(
        r'\b(total|gesamt|summe|amount|subtotal|grand\s*total|ttc|montant)\b',
        re.IGNORECASE
    )
    currency_re = re.compile(
        r'(?:[$£€¥₹₩฿₫₪₺₴₦]|(?:HKD|RMB|RM|MYR|USD|SGD|CHF|EUR|CNY|AUD|CAD|NZD|GBP)\s*)',
        re.IGNORECASE
    )
    price_re = re.compile(r'(\d{1,6}[.,]\d{2})')

    keyword_total = ""
    currency_candidates = []

    for i, line in enumerate(ocr_text.splitlines()):
        prices = price_re.findall(line)
        if not prices:
            continue
        if total_keywords.search(line):
            keyword_total = prices[-1].replace(',', '.')

        elif currency_re.search(line):
            last_price = prices[-1]
            try:
                amount = float(last_price.replace(',', '.'))
                currency_candidates.append((i, amount, last_price))
            except ValueError:
                pass

    if keyword_total:
        return keyword_total

    if currency_candidates:
        # The total is usually the largest amount among currency-sign lines
        _, _, raw = max(currency_candidates, key=lambda x: x[1])
        return raw.replace(',', '.')

    return ""


class NLPProcessor:
    def __init__(
        self,
        model_dir: str, # path to saved MultiTaskReceiptModel checkpoint
        currency_boost:            float = 2.0, # logit multiplier for TOTAL when currency prefix detected
        date_boost:                float = 2.5, # logit multiplier for DATE when date regex matches
        lexicon_weight:            float = 0.65, # weight given to lexicon score (neural gets 1-lexicon_weight)
        cat_confidence_threshold:  float = 0.40, # below this ensemble confidence → default "Shopping"
        device:                    str   = "auto",
    ):
        self.currency_boost           = currency_boost
        self.date_boost               = date_boost
        self.lexicon_weight           = lexicon_weight
        self.cat_confidence_threshold = cat_confidence_threshold

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        log.info(f"Loading NLP model from '{model_dir}' on {self.device} …")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model     = MultiTaskReceiptModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        log.info("Model loaded.")

    def process_image(self, image_path: str) -> Dict:
        raw_text = ocr_image(image_path)
        return self.process_text(raw_text)

    def process_text(self, raw_text: str) -> Dict:
        if not raw_text.strip():
            return self._empty_result(raw_text)

        words   = raw_text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids       = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        ner_logits = out["ner_logits"][0]   # (T, 7)  – still on device
        cat_logits = out["cat_logits"][0]   # (5,)


        # If a sub-token's preceding word starts with a currency symbol,
        # multiply the B-TOTAL (index 5) logit by currency_boost.
        from receipt_dataset import NER_LABEL2ID
        B_TOTAL = NER_LABEL2ID["B-TOTAL"]
        B_DATE  = NER_LABEL2ID["B-DATE"]

        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid >= len(words):
                continue
            word = words[wid]

            if _DATE_PATTERNS.search(word):
                ner_logits[tok_idx, B_DATE] *= self.date_boost

            if _CURRENCY_PREFIX.search(word):
                ner_logits[tok_idx, B_TOTAL] *= self.currency_boost
            elif wid > 0 and _CURRENCY_PREFIX.search(words[wid - 1]):
                ner_logits[tok_idx, B_TOTAL] *= self.currency_boost

        ner_preds = ner_logits.argmax(dim=-1).cpu().numpy()

        word_pred_tags: List[str] = []
        seen_word_ids = set()
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen_word_ids:
                continue
            seen_word_ids.add(wid)
            if wid < len(words):
                word_pred_tags.append(NER_ID2LABEL.get(ner_preds[tok_idx], "O"))

        entities = _bio_to_entities(words[:len(word_pred_tags)], word_pred_tags)

        merchant = entities.get("merchant", "")
        # date_val = entities.get("date", "")
        date_val = _extract_date_heuristic(raw_text)
        if not date_val:
            date_val = entities.get("date", "")

        total_val = entities.get("total","")
        merchant = entities.get("merchant", "")

        # fallbacks
        if not merchant:
            merchant = _extract_merchant_heuristic(raw_text)
        if not total_val:
            total_val = _extract_total_heuristic(raw_text)

        # Neural probability
        neural_probs = torch.softmax(cat_logits, dim=-1).cpu().numpy()

        # Lexicon score (simplified - no WordNet)
        lexicon_probs = lexicon_score(raw_text) 

        # Weighted average
        ensemble     = (1 - self.lexicon_weight) * neural_probs + self.lexicon_weight * lexicon_probs

        best_cat_idx = int(np.argmax(ensemble))
        best_cat_conf = float(ensemble[best_cat_idx])

        if best_cat_conf < self.cat_confidence_threshold:
            category = DEFAULT_CATEGORY
        else:
            category = CATEGORY_ID2LABEL.get(best_cat_idx, DEFAULT_CATEGORY)

        items       = _extract_items_heuristic(raw_text)
        description = _extract_description_heuristic(raw_text, skip_merchant=merchant)

        return {
            "merchant":    merchant,
            "date":        date_val,
            "total":       total_val,
            "items":       items,
            "category":    category,
            "description": description,
            "raw_text":    raw_text,
            "_debug": {
                "neural_category_probs":  {CATEGORY_ID2LABEL[i]: round(float(neural_probs[i]), 4)
                                           for i in range(NUM_CAT_LABELS)},
                "lexicon_category_probs": {CATEGORY_ID2LABEL[i]: round(float(lexicon_probs[i]), 4)
                                           for i in range(NUM_CAT_LABELS)},
                "ensemble_probs":         {CATEGORY_ID2LABEL[i]: round(float(ensemble[i]), 4)
                                           for i in range(NUM_CAT_LABELS)},
                "cat_confidence":         round(best_cat_conf, 4),
                "used_default_category":  best_cat_conf < self.cat_confidence_threshold,
            },
        }

    def _empty_result(self, raw_text: str = "") -> Dict:
        return {
            "merchant":    "",
            "date":        "",
            "total":       "",
            "items":       [],
            "category":    DEFAULT_CATEGORY,
            "description": "",
            "raw_text":    raw_text,
            "_debug":      {},
        }


if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) < 3:
        print("Usage: python nlp_processor.py <model_dir> <image_or_text_path>")
        sys.exit(1)

    model_dir, input_path = sys.argv[1], sys.argv[2]
    processor = NLPProcessor(model_dir)

    if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        result = processor.process_image(input_path)
    else:
        raw = Path(input_path).read_text(encoding="utf-8")
        result = processor.process_text(raw)

    debug = result.pop("_debug", {})
    print("\n----- Result -----")
    pprint.pprint(result)
    print("\n----- Debug -----")
    pprint.pprint(debug)