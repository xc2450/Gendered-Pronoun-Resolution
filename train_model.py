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

test_dist_df = pd.read_csv('test_dist_df.csv')
val_dist_df = pd.read_csv('val_dist_df.csv')
train_dist_df = pd.read_csv('train_dist_df.csv')
test_emb = pd.read_json('contextual_embeddings_gap_test.json')
validation_emb = pd.read_json('contextual_embeddings_gap_validation.json')
train_emb = pd.read_json('contextual_embeddings_gap_train.json')
test_df  = pd.read_table('gap-coreference/gap-development.tsv')
train_df = pd.read_table('gap-coreference/gap-test.tsv')
val_df   = pd.read_table('gap-coreference/gap-validation.tsv')

from keras.layers import *
import keras.backend as K
from keras.models import *
import keras
from keras import optimizers
from keras import callbacks
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

class End2End_NCR():
    
    def __init__(self, word_input_shape, dist_shape, embed_dim=20): 
        
        self.word_input_shape = word_input_shape
        self.dist_shape   = dist_shape
        self.embed_dim    = embed_dim
        self.buckets      = [1, 2, 3, 4, 5, 8, 16, 32, 64] 
        self.hidden_dim   = 150
        
    def build(self):
        
        A, B, P = Input((self.word_input_shape,)), Input((self.word_input_shape,)), Input((self.word_input_shape,))
        dist1, dist2 = Input((self.dist_shape,)), Input((self.dist_shape,))
        inputs = [A, B, P]
        dist_inputs = [dist1, dist2]
        
        self.dist_embed = Embedding(len(self.buckets)+1, self.embed_dim)
        self.ffnn       = Sequential([Dense(self.hidden_dim, use_bias=True),
                                     Activation('relu'),
                                     Dropout(rate=0.2, seed = 7),
                                     Dense(1, activation='linear')])
        
        dist_embeds = [self.dist_embed(dist) for dist in dist_inputs]
        dist_embeds = [Flatten()(dist_embed) for dist_embed in dist_embeds]
        PA = Multiply()([inputs[0], inputs[2]])
        PB = Multiply()([inputs[1], inputs[2]])
        PA = Concatenate(axis=-1)([P, A, PA, dist_embeds[0]])
        PB = Concatenate(axis=-1)([P, B, PB, dist_embeds[1]])
        PA_score = self.ffnn(PA)
        PB_score = self.ffnn(PB)
        score_e  = Lambda(lambda x: K.zeros_like(x))(PB_score)
        output = Concatenate(axis=-1)([PA_score, PB_score, score_e]) 
        output = Activation('softmax')(output)        
        model = Model(inputs+dist_inputs, output)
        
        return model

def create_input(embed_df, dist_df):
    
    assert len(embed_df) == len(dist_df)
    all_P, all_A, all_B = [] ,[] ,[]
    all_label = []
    all_dist_PA, all_dist_PB = [], []
    
    for i in tqdm(range(len(embed_df))):
        
        all_P.append(embed_df.loc[i, "emb_P"])
        all_A.append(embed_df.loc[i, "emb_A"])
        all_B.append(embed_df.loc[i, "emb_B"])
        all_dist_PA.append(dist_df.loc[i, "D_PA"])
        all_dist_PB.append(dist_df.loc[i, "D_PB"])
        label = embed_df.loc[i, "label"]
        if label == "A": 
            all_label.append(0)
        elif label == "B": 
            all_label.append(1)
        else: 
            all_label.append(2)
    
    return [np.asarray(all_A), np.asarray(all_B), np.asarray(all_P),
            np.expand_dims(np.asarray(all_dist_PA),axis=1),
            np.expand_dims(np.asarray(all_dist_PB),axis=1)],all_label

new_emb_df = pd.concat([train_emb, validation_emb])
new_emb_df = new_emb_df.reset_index(drop=True)
new_dist_df = pd.concat([train_dist_df, val_dist_df])
new_dist_df = new_dist_df.reset_index(drop=True)

X_train, y_train = create_input(new_emb_df, new_dist_df)
X_test, y_test = create_input(test_emb, test_dist_df)

min_loss = 1.0
best_model = 0
# Use Kfold to get best model

from sklearn.model_selection import KFold
n_fold = 5
kfold = KFold(n_splits=n_fold, shuffle=True, random_state=3)
for fold_n, (train_index, valid_index) in enumerate(kfold.split(X_train[0])):
    
    X_tr  = [inputs[train_index] for inputs in X_train]
    X_val = [inputs[valid_index] for inputs in X_train]
    y_tr  = np.asarray(y_train)[train_index]
    y_val = np.asarray(y_train)[valid_index]
    
    model = End2End_NCR(word_input_shape=X_train[0].shape[1], dist_shape=X_train[3].shape[1]).build()
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss="sparse_categorical_crossentropy")
    file_path = "best_model_{}.hdf5".format(fold_n+1)
    check_point = callbacks.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_best_only = True, mode = "min")
    early_stop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=100)
    hist = model.fit(X_tr, y_tr, batch_size=128, epochs=1000, validation_data=(X_val, y_val), verbose=0,
              shuffle=True, callbacks = [check_point, early_stop])
    
    if min(hist.history['val_loss']) < min_loss:
        min_loss = min(hist.history['val_loss'])
        best_model = fold_n + 1

model = End2End_NCR(word_input_shape=X_train[0].shape[1], dist_shape=X_train[3].shape[1]).build()
model.load_weights("./best_model_{}.hdf5".format(best_model))
pred = model.predict(x = X_test, verbose = 0)

sub_df = pd.read_csv('sample_submission_stage_1.csv')
sub_df.loc[:, 'A'] = pd.Series(pred[:, 0])
sub_df.loc[:, 'B'] = pd.Series(pred[:, 1])
sub_df.loc[:, 'NEITHER'] = pd.Series(pred[:, 2])


from sklearn.metrics import log_loss
y_one_hot = np.zeros((2000, 3))
for i in range(len(y_test)):
    y_one_hot[i, y_test[i]] = 1
log_loss(y_one_hot, pred)

sub_df.to_csv("submission.csv", index=False)

from sklearn.metrics import f1_score
y_true = np.argmax(y_one_hot, axis = 1)
y_pred = np.argmax(pred, axis = 1)
print("Overall F1: ", f1_score(y_true, y_pred, average = 'macro'))

f_pred = []
m_pred = []
f_true = []
m_true = []
for index, row in test_df.iterrows():
    if row['Pronoun'] == 'her' or row['Pronoun'] == 'Her':
        f_pred.append(y_pred[index])
        f_true.append(y_true[index])
    elif row['Pronoun'] == 'his' or row['Pronoun'] == 'His':
        m_pred.append(y_pred[index])
        m_true.append(y_true[index])
female = f1_score(f_true, f_pred, average = 'macro')
male = f1_score(m_true, m_pred, average = 'macro')
print("Female F1: ", female)
print("Male F1: ", male)
print("Bias: ", female/male)
