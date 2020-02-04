# from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model 
import string
import random
import pickle


os.environ["CUDA_VISIBLE_DEVICES"]="-1"  


BATCH_SIZE = 64
NUM_EPOCHS = 3



# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, vocab):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

def get_data(data_dir, dataset_id, vocab, tokenizer, is_train=True):
    if(is_train) :
        positive_docs = process_docs(data_dir + '/train/pos', vocab)
        negative_docs = process_docs(data_dir + '/train/neg', vocab)

        num_rows = len(positive_docs)
        num_divisions = 5
        division_len = num_rows // num_divisions
        start_ind = (division_len * (dataset_id-1))
        end_ind = (division_len * dataset_id)
        positive_docs = positive_docs[start_ind : end_ind]

        num_rows = len(negative_docs)
        num_divisions = 5
        division_len = num_rows // num_divisions
        start_ind = (division_len * (dataset_id-1))
        end_ind = (division_len * dataset_id)
        negative_docs = negative_docs[start_ind : end_ind]


    else:
        positive_docs = process_docs(data_dir + '/test/pos', vocab)
        negative_docs = process_docs(data_dir + '/test/neg', vocab)


    docs = negative_docs + positive_docs

    positive_docs_len = len(positive_docs)
    negative_docs_len = len(negative_docs)

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(docs)

    max_length = 80
    X = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # define training labels
    y = np.array([0 for _ in range(negative_docs_len)] + [1 for _ in range(positive_docs_len)])

    return X, y

def train_on_device(data_dir, dataset_id, model_path, ckpt_path, weight_updates_path):
    """
    Returns (n, weight_updates) after training on local data    
    """
    
    # Store pre-trained model and weights 
    old_model = load_model(model_path)
    old_model.load_weights(ckpt_path)
    
    # Initialize model and checkpoint, which are obtained from server
    device_model = load_model(model_path)
    device_model.load_weights(ckpt_path)
    
#    print(device_model.summary())

    # load the vocabulary
    vocab_filename = data_dir + '/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    
    # create the tokenizer
    tokenizer = Tokenizer()

    # loading tokenizer from file
    tokenizer_filename = data_dir + 'tokenizer.pickle'
    with open(tokenizer_filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Get training data present on device
    X_train, y_train = get_data(data_dir, dataset_id, vocab, tokenizer, is_train=True)

    X_test, y_test = get_data(data_dir, dataset_id, vocab, tokenizer, is_train=False)

    scores = device_model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy(before training): %.2f%%" % (scores[1]*100))

    # Train model
    device_model.fit(X_train, y_train, 
    epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE)

    
    
    scores = device_model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy(after training): %.2f%%" % (scores[1]*100))

    # Load model to store weight updates
    weight_updates = load_model(model_path)
    
    # Number of batches trained on device
    num_batches = X_train.shape[0] // BATCH_SIZE
    
    # Calculate weight updates
    for i in range(len(device_model.layers)):
        
        # Pre-trained weights
        old_layer_weights = old_model.layers[i].get_weights()
        
        # Post-trained weights        
        new_layer_weights = device_model.layers[i].get_weights()

        # Weight updates calculation
        weight_updates.layers[i].set_weights(num_batches * (np.asarray(new_layer_weights) - np.asarray(old_layer_weights)))
    
    #    print("old weights: ",  old_layer_weights)
    #    print("new weights: ",  new_layer_weights)
    #    print("weight updates: ",  weight_updates.layers[i].get_weights())
    
    
    # Save weight updates
    weight_updates.save_weights(weight_updates_path)
    
    return (num_batches, weight_updates_path)
