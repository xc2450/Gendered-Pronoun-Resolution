import numpy as np
import pandas as pd 
import os
import zipfile
import gc
from tqdm import tqdm_notebook as tqdm
import re
import modeling
import extract_features
import tokenization
import tensorflow as tf
import spacy
import torch as torch
from sklearn.metrics import f1_score
import pprint as pprint
from pytorch_pretrained_bert2 import BertTokenizer, BertModel, BertForMaskedLM
import math
nlp = spacy.load('en_core_web_lg')

# load the dataset
test_df  = pd.read_table('gap-coreference/gap-development.tsv')
train_df = pd.read_table('gap-coreference/gap-test.tsv')
val_df   = pd.read_table('gap-coreference/gap-validation.tsv')

def count_char(text, offset):   
    count = 0
    for pos in range(offset):
        if text[pos] != " ": count +=1
    return count

def candidate_length(candidate):
    count = 0
    for i in range(len(candidate)):
        if candidate[i] !=  " ": count += 1
    return count

def count_token_length_special(token):
    count = 0
    special_token = ["#", " "]
    if token == '[SEP]':
        return count
    if token == '#':
        return 1
    for i in range(len(token)):
        if token[i] not in special_token: count+=1
    return count

def process_text(df, tokenizer):
    processed_text = []
    labels = []
    targets = []
    aligns = []
    for i in df['Text']:
        passage_text_processed = nlp(i)
        passage_with_separators = ' '.join(['[CLS]'] + [sent.text + ' [SEP]' for sent in passage_text_processed.sents])
        passage_with_separators_tokenized = tokenizer.tokenize(passage_with_separators)
        align = [(-1,-1)]
        count = 0
        for j in range(1, len(passage_with_separators_tokenized)):
            length = count_token_length_special(passage_with_separators_tokenized[j])
            align.append((count, count+length))
            count += length 
        aligns.append(align)
        indexed_tokens = tokenizer.convert_tokens_to_ids(passage_with_separators_tokenized)
        processed_text.append(indexed_tokens)
    for index, row in df.iterrows():
        if row['A-coref'] ==  True:
            labels.append((row['A-offset'], row['A-offset']+len(row['A'])))
        elif row['B-coref'] == True:
            labels.append((row['B-offset'], row['B-offset']+len(row['B'])))
        else:
            labels.append("")
        targets.append(row['Pronoun-offset'])
    return processed_text, labels, targets, aligns


def reverse_index(l):
    dicts = {}
    for index, i in enumerate(l):
        dicts[i[0]] = index 
    return dicts

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train, y_train, target_train, aligns_train = process_text(train_df,tokenizer)
X_test, y_test, target_test, aligns_test= process_text(test_df,tokenizer)
X_val, y_val, target_val, aligns_val = process_text(val_df,tokenizer)

aligns_train_ = [reverse_index(i) for i in aligns_train]
aligns_test_ = [reverse_index(i) for i in aligns_test]
aligns_val_ = [reverse_index(i) for i in aligns_val]

model = BertModel.from_pretrained('bert-base-uncased')

def get_attention(df, X_train, dictionary, model):
    att = []
    with torch.no_grad():
        for index, i in enumerate(X_train):
            P_char_start = count_char(df.loc[index, 'Text'], df.loc[index, 'Pronoun-offset'])
            if index%100 == 0:
                print(index)
            tokens_tensor = torch.tensor([i])
            encoded_layers, _,attention_layers = model(tokens_tensor, output_all_encoded_layers=True)
            probs = [layer['attn_probs'].numpy()[0,:,dictionary[index][P_char_start],:] for layer in attention_layers]
            att.append(probs)
    return att

val_att = get_attention(val_df, X_val, aligns_val_, model)        
train_att = get_attention(train_df, X_train, aligns_train_, model) 

def get_alignment(start, end, aligns, text):
    l = []
    start_index = count_char(text, start)
    end_index = count_char(text, end)
    for index, i in enumerate(aligns):
        if end_index > i[0] >= start_index:
            l.append(index)
    return l

def extract_named_entities(df, aligns):
    named_entities = []
    for index, i in enumerate(df['Text']):  
        doc = nlp(i)
        named_entity = []
        alignment = []
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                named_entity.append((ent.text, ent.start_char, ent.end_char))
                alignment.append(get_alignment(ent.start_char, ent.end_char, aligns[index], i))
        named_entities.append([named_entity, alignment])
    return named_entities

val_entities = extract_named_entities(val_df, aligns_val)
train_entities = extract_named_entities(train_df, aligns_train)
test_entities = extract_named_entities(test_df, aligns_test)

def predict(att_probs, entities):
    pred = np.zeros((12,12,len(entities),2))
    for k in range(len(entities)):
        for i in range(12):
            for j in range(12):
                max_prob = 0
                max_indices = [0,0]
                for l in range(len(entities[k][0])):
                    indices = entities[k][1][l]
                    prob = np.sum(att_probs[k][i][j][indices])
                    if prob>max_prob:
                        max_prob = prob
                        max_indices = [entities[k][0][l][1], entities[k][0][l][2]]
                    pred[i][j][k][0] = max_indices[0]
                    pred[i][j][k][1] = max_indices[1]
    return pred

val_pred = predict(val_att, val_entities)
train_pred = predict(train_att, train_entities)
test_pred = predict(test_att, test_entities)

def evaluate(pred, df):
    prediction = np.zeros((12,12,len(df)))
    labels = np.zeros(len(df))
    for i in range(len(df)):
        if df.loc[i, 'A-coref'] == True:
            labels[i]=0
        elif df.loc[i, 'B-coref'] == True:
            labels[i]=1
        else:
            labels[i] =2
    for i in range(len(df)):
        A = df.loc[i, 'A-offset']
        B = df.loc[i,'B-offset']
        for j in range(12):
            for k in range(12):
                if pred[j][k][i][0] <= A <= pred[j][k][i][1]:
                    prediction[j][k][i] = 0
                elif pred[j][k][i][0]<= B <= pred[j][k][i][1]:
                    prediction[j][k][i] = 1
                else:
                    prediction[j][k][i] = 2
    f1_matrix = np.zeros((12,12))
    for i in range(12):
        for j in range(12):
            f1_matrix[i][j] = f1_score(labels, prediction[i][j],average='macro')
    return f1_matrix,prediction

f1_matrix,prediction = evaluate(val_pred, val_df)