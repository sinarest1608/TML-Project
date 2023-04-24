# import numpy as np
# from tensorflow.keras.datasets import cifar100
# import tensorflow_datasets as tfds
# from sklearn.model_selection import train_test_split
# import pickle

# # Load CIFAR-100 dataset
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# # Split the training data into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# # Load the class names from the CIFAR-100 dataset
# class_labels = tfds.load('cifar100', split='train', batch_size=-1)['label'].numpy()
# class_labels = np.unique(class_labels)

# # Convert the labels to integer indices
# y_train = y_train.flatten()
# y_val = y_val.flatten()
# y_test = y_test.flatten()

# # Save the dataset and labels to a .npz file
# np.savez('data_cifar100.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, class_labels=class_labels)


# import scipy.io as sio
# import os

# # Define the local directory where you want to save the dataset
# local_dir = "svhn_data/"

# # Create the local directory if it doesn't exist
# if not os.path.exists(local_dir):
#     os.makedirs(local_dir)

# # Download the train dataset
# train_data = sio.loadmat('http://ufldl.stanford.edu/housenumbers/train_32x32.mat')

# # Save the train dataset to the local directory
# sio.savemat(os.path.join(local_dir, 'train_32x32.mat'), train_data)

# # Download the test dataset
# test_data = sio.loadmat('http://ufldl.stanford.edu/housenumbers/test_32x32.mat')

# # Save the test dataset to the local directory
# sio.savemat(os.path.join(local_dir, 'test_32x32.mat'), test_data)


import numpy as np
import urllib.request
import os

# Define the local directory where you want to save the dataset
local_dir = "svhn_data/"

# Create the local directory if it doesn't exist
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# Download the train dataset
train_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
train_path = os.path.join(local_dir, 'train_32x32.mat')
urllib.request.urlretrieve(train_url, train_path)

# Download the test dataset
test_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
test_path = os.path.join(local_dir, 'test_32x32.mat')
urllib.request.urlretrieve(test_url, test_path)

# Load the dataset and save it in .npz format
train_data = sio.loadmat(train_path)
X_train = np.transpose(train_data['X'], (3, 0, 1, 2))
y_train = train_data['y'].flatten() - 1  # convert labels to 0-9 range

test_data = sio.loadmat(test_path)
X_test = np.transpose(test_data['X'], (3, 0, 1, 2))
y_test = test_data['y'].flatten() - 1  # convert labels to 0-9 range

# Save the dataset to the local directory in .npz format
np.savez(os.path.join(local_dir, 'svhn_data.npz'), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

