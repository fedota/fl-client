from __future__ import absolute_import, division, print_function, unicode_literals

import gzip
import os
import sys
import time
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf
import numpy as np

import six
from pathlib import Path

import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model


SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.



def train_on_device(device_path, model_path, ckpt_path):
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
    
    # Get training data present on device
    train_images, train_labels = get_mnist_images_and_labels(device_path)
    
    # Train model
    device_model.fit(train_images, train_labels,
          batch_size=BATCH_SIZE,
          epochs=2,
          verbose=1)
    
    # Load model to store weight updates
    update_weights = load_model(model_path)
    
    # Number of batches trained on device
    num_batches = train_images.shape[0] // BATCH_SIZE
    
    # Calculate weight updates
    for i in range(len(device_model.layers)):
        
        # Pre-trained weights
        old_layer_weights = old_model.layers[i].get_weights()
        
        # Post-trained weights        
        new_layer_weights = device_model.layers[i].get_weights()

        # Weight updates calculation
        update_weights.layers[i].set_weights(num_batches * (np.asarray(new_layer_weights) - np.asarray(old_layer_weights)))
    
    #    print("old weights: ",  old_layer_weights)
    #    print("new weights: ",  new_layer_weights)
    #    print("update weights: ",  update_weights.layers[i].get_weights())
    
    
    # Save weight updates
    update_weights_path = device_path + '/weight_updates'
    update_weights.save_weights(update_weights_path)
    
    return (num_batches, update_weights_path)
