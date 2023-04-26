import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import os
import cv2
from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model('./cifar100_cnn.h5')

epsilon = 0.01

@tf.function
def fgsm(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.keras.losses.categorical_crossentropy(y, model(x))
    grad = tape.gradient(loss, x)
    signed_grad = tf.sign(grad)
    return x + epsilon * signed_grad


num_classes = 100
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


adv_x_test = np.zeros_like(x_test)
for i in range(len(x_test)):
    adv_x_test[i] = fgsm(tf.convert_to_tensor(x_test[i][None, ...]), 
                         tf.convert_to_tensor(y_test[i][None, ...]))[0]

if not os.path.exists('adversarial_examples'):
    os.makedirs('adversarial_examples')

for i in range(len(adv_x_test)):
 
    img = adv_x_test[i]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    filename = f"adversarial_examples/{i}.png"
    cv2.imwrite(filename, img)
