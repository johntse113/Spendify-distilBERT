import os
import re
import json
import difflib
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("pytesseract / Pillow not installed – OCR unavailable.")


# !!! path of tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not Path(pytesseract.pytesseract.tesseract_cmd).exists():
    raise FileNotFoundError(f"Tesseract not found at: {pytesseract.pytesseract.tesseract_cmd}")

NER_LABEL2ID: Dict[str, int] = {
    "O":          0,
    "B-MERCHANT": 1,
    "I-MERCHANT": 2,
    "B-DATE":     3,
    "I-DATE":     4,
    "B-TOTAL":    5,
    "I-TOTAL":    6,
}
NER_ID2LABEL: Dict[int, str] = {v: k for k, v in NER_LABEL2ID.items()}

CATEGORY_LABEL2ID: Dict[str, int] = {
    "Food & Dining":  0,
    "Transport":      1,
    "Shopping":       2,
    "Entertainment":  3,
    "Utilities":      4,
}
CATEGORY_ID2LABEL: Dict[int, str] = {v: k for k, v in CATEGORY_LABEL2ID.items()}
DEFAULT_CATEGORY = "Shopping"

LABEL_FIELD_MAP = {
    "company": ("B-MERCHANT", "I-MERCHANT"),
    "date":    ("B-DATE",     "I-DATE"),
    "total":   ("B-TOTAL",    "I-TOTAL"),
}

MODEL_NAME   = "distilbert-base-uncased"
MAX_SEQ_LEN  = 256


def preprocess_image(image_path: str):
    img = Image.open(image_path).convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def ocr_image(image_path: str) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract pillow")
    img = preprocess_image(image_path)
    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()

# Fuzzy substring search to align labels with OCR text
def fuzzy_find_span(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    if not needle:
        return None

    lower_hay  = haystack.lower()
    lower_need = needle.lower()
    idx = lower_hay.find(lower_need)
    if idx != -1:
        return (idx, idx + len(needle))
    n = len(needle)
    best_ratio, best_pos = 0.0, None
    step = max(1, n // 4)
    for start in range(0, len(haystack) - n + 1, step):
        window = haystack[start : start + n]
        ratio  = difflib.SequenceMatcher(None, lower_need, window.lower()).ratio()
        if ratio > best_ratio:
            best_ratio, best_pos = ratio, start

    if best_ratio >= 0.70 and best_pos is not None:
        return (best_pos, best_pos + n)
    return None


def word_level_bio_labels(text: str, labels: Dict[str, str]) -> Tuple[List[str], List[str]]:
    words = text.split()
    tags  = ["O"] * len(words)
    char_tags = ["O"] * len(text)

    for field, value in labels.items():
        if field not in LABEL_FIELD_MAP or not value:
            continue
        b_tag, i_tag = LABEL_FIELD_MAP[field]
        span = fuzzy_find_span(text, str(value))
        if span is None:
            continue
        start, end = span
        char_tags[start] = b_tag
        for i in range(start + 1, end):
            char_tags[i] = i_tag

    pos = 0
    for w_idx, word in enumerate(words):
        start = text.find(word, pos)
        if start == -1:
            pos += len(word) + 1
            continue
        end = start + len(word)
        span_tags = char_tags[start:end]

        for t in span_tags:
            if t != "O":
                tags[w_idx] = t
                break
        pos = end

    return words, tags




class NERDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: DistilBertTokenizerFast,
        max_len: int = MAX_SEQ_LEN,
        ocr_cache_dir: Optional[str] = None,
    ):
        self.tokenizer       = tokenizer
        self.max_len         = max_len
        self.ocr_cache_dir   = ocr_cache_dir
        self.samples: List[Tuple[str, Dict]] = []   # (ocr_text, labels_dict)

        data_path = Path(data_dir)
        jpg_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.JPG"))

        for jpg_path in jpg_files:
            txt_path = jpg_path.with_suffix(".txt")
            if not txt_path.exists():
                logging.warning(f"Label file not found for {jpg_path.name}; skipping.")
                continue

            ocr_text = self._get_ocr_text(jpg_path)
            if not ocr_text:
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            try:
                label_json = json.loads(content)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in {txt_path.name}; skipping.")
                continue

            self.samples.append((ocr_text, label_json))

        logging.info(f"NERDataset: loaded {len(self.samples)} samples from {data_dir}")


    def _get_ocr_text(self, jpg_path: Path) -> str:
        if self.ocr_cache_dir:
            cache_file = Path(self.ocr_cache_dir) / (jpg_path.stem + ".ocr.txt")
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8")

        text = ocr_image(str(jpg_path))

        if self.ocr_cache_dir:
            Path(self.ocr_cache_dir).mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text, encoding="utf-8")
        return text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ocr_text, label_json = self.samples[idx]

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

        aligned_labels = []
        prev_word_id   = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(NER_LABEL2ID.get(word_tags[word_id], 0))
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }


class CategoryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: DistilBertTokenizerFast,
        max_len: int = MAX_SEQ_LEN,
        ocr_cache_dir: Optional[str] = None,
    ):
        self.tokenizer     = tokenizer
        self.max_len       = max_len
        self.samples: List[Tuple[str, int]] = []   # (ocr_text, category_id)
        ner_ds = NERDataset.__new__(NERDataset)
        ner_ds.tokenizer     = tokenizer
        ner_ds.max_len       = max_len
        ner_ds.ocr_cache_dir = ocr_cache_dir
        ner_ds.samples       = []
        ner_ds._get_ocr_text = lambda p: NERDataset._get_ocr_text(ner_ds, p)

        data_path = Path(data_dir)
        jpg_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.JPG"))

        for jpg_path in jpg_files:
            txt_path = jpg_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            ocr_text = ner_ds._get_ocr_text(jpg_path)
            if not ocr_text:
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            try:
                label_json = json.loads(content)
            except json.JSONDecodeError:
                continue

            raw_cat = label_json.get("category", DEFAULT_CATEGORY)
            cat_id  = CATEGORY_LABEL2ID.get(raw_cat, CATEGORY_LABEL2ID[DEFAULT_CATEGORY])
            self.samples.append((ocr_text, cat_id))

        logging.info(f"CategoryDataset: loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ocr_text, cat_id = self.samples[idx]
        encoding = self.tokenizer(
            ocr_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(cat_id, dtype=torch.long),
        }
