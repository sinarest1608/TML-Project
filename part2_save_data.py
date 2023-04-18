import numpy as np
from tensorflow.keras.datasets import cifar100
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import pickle

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Load the class names from the CIFAR-100 dataset
class_labels = tfds.load('cifar100', split='train', batch_size=-1)['label'].numpy()
class_labels = np.unique(class_labels)

# Convert the labels to integer indices
y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()

# Save the dataset and labels to a .npz file
np.savez('data_cifar100.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, class_labels=class_labels)
