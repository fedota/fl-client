"""Simulates training on client devices."""
import os

import numpy as np
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


SOURCE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
WORK_DIRECTORY = "data"
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


def get_mnist_images_and_labels(data_dir):
    """Loads the labelled images."""
    images = np.load(data_dir + "/mnist-images.npy")
    labels = np.load(data_dir + "/mnist-labels.npy")
    return (images, labels)


def train_on_device(data_dir, dataset_id, model_path, ckpt_path, weight_updates_path):
    """Returns (n, weight_updates) after training on local data."""
    # Store pre-trained model and weights
    old_model = load_model(model_path)
    old_model.load_weights(ckpt_path)

    # Initialize model and checkpoint, which are obtained from server
    device_model = load_model(model_path)
    device_model.load_weights(ckpt_path)

    #    print(device_model.summary())

    # Get training data present on device
    train_images, train_labels = get_mnist_images_and_labels(data_dir)

    NUM_IMAGES = train_images.shape[0]
    NUM_DIVISIONS = 5
    DIVISION_LEN = NUM_IMAGES // NUM_DIVISIONS

    start_ind = DIVISION_LEN * (dataset_id - 1)
    end_ind = DIVISION_LEN * dataset_id
    train_images = train_images[start_ind:end_ind]
    train_labels = train_labels[start_ind:end_ind]

    # Train model
    device_model.fit(
        train_images, train_labels, batch_size=BATCH_SIZE, epochs=1, verbose=1,
    )

    # Load model to store weight updates
    weight_updates = load_model(model_path)

    # Number of batches trained on device
    num_batches = train_images.shape[0] // BATCH_SIZE

    # Calculate weight updates
    for i in range(len(device_model.layers)):

        # Pre-trained weights
        old_layer_weights = old_model.layers[i].get_weights()

        # Post-trained weights
        new_layer_weights = device_model.layers[i].get_weights()

        # Weight updates calculation
        weight_updates.layers[i].set_weights(
            num_batches
            * (np.asarray(new_layer_weights) - np.asarray(old_layer_weights)),
        )

    #    print("old weights: ",  old_layer_weights)
    #    print("new weights: ",  new_layer_weights)
    #    print("weight updates: ",  weight_updates.layers[i].get_weights())

    # Save weight updates
    weight_updates.save_weights(weight_updates_path)

    return (num_batches, weight_updates_path)
