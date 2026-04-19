# Spendify-distilBERT
This is the repo for the distilBERT NLP model for the Spendify project.



# How to run:
```
pip install -r requirements.txt
```
```
python nlp_processor_skip.py "./saved_model_kfold" <image path>
```

# How to train:
Store the training samples somewhere (e.g. ./data/all), then:
```
 python train_kfold.py --data_dir ./data/all --output_dir ./saved_model_kfold --k_folds 5 --epochs 20 --batch_size 8
```
