# Text Classification
BERT based Text Classification

Use "Text_Classification.ipynb" to run in google colab.

# Requirments
toch 1.6.0

transformers 3.0.2

python 3.6.9

# Usages
1. Data pre-processing

Pre-process the data and split dataset into train and test sets

python process_data.py

2. Train Model

Test classification task is based on BERT model. BertForSequenceClassification is used and "dbmdz/bert-base-turkish-cased" is utilized as pre-trained BERT model for Turkish.

python model.py

3. Test Model

Saved model is used to test the test dataset. 

python test.py
