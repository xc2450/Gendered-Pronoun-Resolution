# Gendered-Pronoun-Resolution
We did not include BERT model because of its large size. We also included GAP coreference dataset and our generated prediction for the test set. 

feature_extraction.py extracts contexual embedding and distance features from the dataset. 

train_model.py trains the mention-ranking model, save the prediction for test set, and print out F1 score. 

BERT_heuristic.py make prediction for each layer and head and print out the F1 matrix