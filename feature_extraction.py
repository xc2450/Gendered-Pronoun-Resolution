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
nlp = spacy.load('en_core_web_lg')

def extract_dist_features(df):
    # extract distance features
    index = df.index
    columns = ["D_PA", "D_PB", "IN_URL"]
    dist_df = pd.DataFrame(index = index, columns = columns)
    for i in tqdm(range(len(df))):
        text = df.loc[i, 'Text']
        P_offset = df.loc[i,'Pronoun-offset']
        A_offset = df.loc[i, 'A-offset']
        B_offset = df.loc[i, 'B-offset']
        P, A, B  = df.loc[i,'Pronoun'], df.loc[i, 'A'], df.loc[i, 'B']
        URL = df.loc[i, 'URL']
        dist_df.iloc[i] = distance_features(P, A, B, P_offset, A_offset, B_offset, text, URL)
    return dist_df


def bs(lens, target):
    # run binary search to find the target
    low, high = 0, len(lens) - 1
    while low < high:
        mid = low + int((high - low) / 2)
        if target > lens[mid]:
            low = mid + 1
        elif target < lens[mid]:
            high = mid
        else:
            return mid + 1
    return low

def bin_distance(dist):
    # calculate the distance between the target and the reference
    buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]  
    low, high = 0, len(buckets)
    while low < high:
        mid = low + int((high-low) / 2)
        if dist > buckets[mid]:
            low = mid + 1
        elif dist < buckets[mid]:
            high = mid
        else:
            return mid
    return low

def distance_features(P, A, B, char_offsetP, char_offsetA, char_offsetB, text, URL):
    # Compute the distance between A, B and P, and whether the URL contains A or B
    doc = nlp(text)
    lens = [token.idx for token in doc]
    mention_offsetP = bs(lens, char_offsetP) - 1
    mention_offsetA = bs(lens, char_offsetA) - 1
    mention_offsetB = bs(lens, char_offsetB) - 1
    mention_distA = mention_offsetP - mention_offsetA 
    mention_distB = mention_offsetP - mention_offsetB
    splited_A = A.split()[0].replace("*", "")
    splited_B = B.split()[0].replace("*", "")
    if re.search(splited_A[0], str(URL)):
        contains = 0
    elif re.search(splited_B[0], str(URL)):
        contains = 1
    else:
        contains = 2
    dist_binA = bin_distance(mention_distA)
    dist_binB = bin_distance(mention_distB)
    output =  [dist_binA, dist_binB, contains]
    return output

test_dist_df = extract_dist_features(test_df)
test_dist_df.to_csv('test_dist_df.csv', index=False)
val_dist_df = extract_dist_features(val_df)
val_dist_df.to_csv('val_dist_df.csv', index=False)
train_dist_df = extract_dist_features(train_df)
train_dist_df.to_csv('train_dist_df.csv', index=False)


def get_surrounding_emb(start, end, features):
    emb = np.zeros(6144)
    # get prev 2 and next 2 embeddings
    prev_2 = start - 2
    if prev_2 < 0:
        prev_2 = 0
    next_2 = end + 2
    if next_2 > len(features) - 1:
        next_2 = len(features) - 1
    emb[:1024] = np.asarray(features[prev_2]["layers"][0]['values'])
    emb[1024:2048] = np.asarray(features[prev_2+1]["layers"][0]['values'])
    emb[2048:3072] = np.asarray(features[next_2]["layers"][0]['values'])
    emb[3072:4096] = np.asarray(features[next_2-1]["layers"][0]['values'])
    # get average of prev 5 and next 5 embeddings
    prev_5 = start - 5
    if prev_5 < 0:
        prev_5 = 0
    next_5 = end + 5
    if next_5 > len(features) - 1:
        next_5 = len(features) - 1
    for i in range(prev_5, start):
        emb[4096:5120] += np.asarray(features[i]["layers"][0]['values'])
    emb[4096:5120] /= start-prev_5  
    for i in range(end, next_5):
        emb[5120:6144] += np.asarray(features[i]["layers"][0]['values'])
    emb[5120:6144] /= next_5-end  
    return emb

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
    for i in range(len(token)):
        if token[i] not in special_token: count+=1
    return count


def embed_by_bert(df):    
    '''
    Runs a forward propagation of BERT on input text, extracting contextual word embeddings
    Input: data, a pandas DataFrame containing the information in one of the GAP files

    Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun. Each embedding is a numpy array of shape (768)
    columns: "emb_A": the embedding for word A
             "emb_B": the embedding for word B
             "emb_P": the embedding for the pronoun
             "label": the answer to the coreference problem: "A", "B" or "NEITHER"
    '''
    text = df['Text']
    text.to_csv('input.txt', index=False, header=False)
# run BERT base on the input text

    os.system("python3 extract_features.py \
               --input_file=input.txt \
               --output_file=output.jsonl \
               --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
               --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
               --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
               --layers=-1 \
               --max_seq_length=256 \
               --batch_size=8")

# Run BERT large to extract embedding

#     os.system("python3 extract_features.py \
#                --input_file=input.txt \
#                --output_file=output.jsonl \
#                --vocab_file=uncased_L-24_H-1024_A-16/vocab.txt \
#                --bert_config_file=uncased_L-24_H-1024_A-16/bert_config.json \
#                --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
#                --layers=-1 \
#                --max_seq_length=256 \
#                --batch_size=8")
    
    bert_output = pd.read_json("output.jsonl", lines = True)
    bert_output.head()
    
    features = bert_output.loc[0, "features"]
    os.system("rm input.txt")
    os.system("rm output.json1")
    index = df.index
    columns = ["emb_A", "emb_A_s", "emb_B", "emb_B_s", "emb_P","emb_P_s", "label"]
    emb = pd.DataFrame(index = index, columns = columns)
    emb.index.name = "ID"
    
    for i in tqdm(range(len(text))):
        
        features = bert_output.loc[i, "features"]
        P_char_start = count_char(df.loc[i, 'Text'], df.loc[i, 'Pronoun-offset'])
        A_char_start = count_char(df.loc[i, 'Text'], df.loc[i, 'A-offset'])
        B_char_start = count_char(df.loc[i, 'Text'], df.loc[i, 'B-offset'])
        A_length = candidate_length(df.loc[i, 'A'])
        B_length = candidate_length(df.loc[i, 'B'])
            
        emb_A = np.zeros(1024)
        emb_B = np.zeros(1024)
        emb_P = np.zeros(1024)
        
        char_count = 0
        cnt_A, cnt_B = 0, 0
        start_A, end_A = -1,-1
        start_B, end_B = -1,-1
        index_P = -1
        
        for j in range(2, len(features)):
            token = features[j]["token"]
            token_length = count_token_length_special(token)
            if char_count == P_char_start:
                emb_P += np.asarray(features[j]["layers"][0]['values']) 
                index_P = j
            if char_count in range(A_char_start, A_char_start+A_length):
                emb_A += np.asarray(features[j]["layers"][0]['values'])
                cnt_A += 1
                if start_A == -1:
                    start_A = j
                end_A = j
            if char_count in range(B_char_start, B_char_start+B_length):
                emb_B += np.asarray(features[j]["layers"][0]['values'])
                cnt_B += 1   
                if start_B == -1:
                    start_B = j
                end_B = j
            char_count += token_length
        
        emb_A /= cnt_A
        emb_B /= cnt_B
        
        emb_A_s = get_surrounding_emb(start_A, end_A, features)
        emb_B_s = get_surrounding_emb(start_B, end_B, features)
        emb_P_s = get_surrounding_emb(index_P, index_P, features)
        
        # Get label
        label = "Neither"
        if (df.loc[i,"A-coref"] == True):
            label = "A"
        if (df.loc[i,"B-coref"] == True):
            label = "B"

        emb.iloc[i] = [emb_A,emb_A_s, emb_B, emb_B_s, emb_P, emb_P_s, label]
        
    return emb    

test_emb = embed_by_bert(test_df)
test_emb.to_json("contextual_embeddings_gap_test.json", orient = 'columns')
validation_emb = embed_by_bert(val_df)
validation_emb.to_json("contextual_embeddings_gap_validation.json", orient = 'columns')
train_emb = embed_by_bert(train_df)
train_emb.to_json("contextual_embeddings_gap_train.json", orient = 'columns')