# Spendify-distilBERT
This is the repo for the multi-task distilBERT NLP model with pytesseract integration for receipt scanning the Spendify project.

This model takes receipt image as the input, and returns with the transaction details in json format.

# Installation:
macOS (Homebrew)
```
brew install tesseract
```
Ubuntu/Debian
```
sudo apt-get install tesseract-ocr
```
Windows\
https://github.com/UB-Mannheim/tesseract/wiki

<br><br>
Install requirements:
```
pip install -r requirements.txt
```

# How to run:
```
python nlp_processor_skip.py "./saved_model_kfold" <image path>
```

# How to train:
Store the training samples somewhere (e.g. ./data/all), then:
```
 python train_kfold.py --data_dir ./data/all --output_dir ./saved_model_kfold --k_folds 5 --epochs 20 --batch_size 8
```
Each training sample must be labeled with a txt file in json format, e.g.
```
{
  "company":  "S.H.H. MOTOR ...",
  "date":     "23-01-2019",
  "address":  "NO. 343, JALAN ...",
  "total":    "20.00",
  "category": "Shopping"
}
```
