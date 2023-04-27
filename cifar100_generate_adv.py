import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import os
import cv2
from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data()


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
model = tf.keras.models.load_model('./cifar100_cnn_new.h5')


frac = 0.05
fgsm_set_x = tf.concat([x_train[:int(frac*x_train.shape[0])],x_test[:int(frac*x_test.shape[0])]],axis=0)
fgsm_set_y = tf.concat([y_train[:int(frac*y_train.shape[0])],y_test[:int(frac*y_test.shape[0])]],axis=0)

from PIL import Image
import numpy as np

@tf.function
def fgsm(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, model(x))
        grad = tape.gradient(loss, x)
    grad = grad/tf.reduce_max(grad)
    # signed_grad = tf.sign(grad)
    return x + (epsilon * grad)



epsilon = 20
batch_size = 128
epochs = 1
benign_acc = tf.keras.metrics.Accuracy()
benign_acc.reset_state()
adversary_acc = tf.keras.metrics.Accuracy()
adversary_acc.reset_state()


adv_x = []
adv_y = []
benign_x = []
benign_y = []
label = []

for epoch in range(epochs):
    prev_i = 0
    for i in range(batch_size,fgsm_set_x.shape[0],batch_size):
        X = fgsm_set_x[prev_i:i]
        y = fgsm_set_y[prev_i:i]    
        benign_x.append(np.uint8(X.numpy().copy()))        
        y_pred = tf.argmax(model.predict(X),axis=1)
        benign_y.append(y_pred)
        benign_acc.update_state(y[:,0],y_pred)
        
        X = fgsm(X,y)
        adv_x.append(np.uint8(X.numpy()))
        y_pred = tf.argmax(model.predict(X),axis=1)
        adv_y.append(y_pred)
        label.append(y)
        adversary_acc.update_state(y[:,0],y_pred)
        
        
        prev_i = i        

        img = Image.fromarray(np.uint8(X[0].numpy()))
        img.save('temp/'+str(epoch)+str(i)+'.png')
    
    
    print(adversary_acc.result(),benign_acc.result())
    adversary_acc.reset_state()
    benign_acc.reset_state()


    
adv_x = np.concatenate(adv_x)
adv_y = np.concatenate(adv_y)
benign_y = np.concatenate(benign_y)
benign_x = np.concatenate(benign_x)
label = np.concatenate(label)
acc = tf.keras.metrics.Accuracy()
# print(acc(label[:,0],benign_y))

# print(np.sum((label[:,0] == benign_y).astype(int)))
# print(np.sum((label[:,0] == adv_y).astype(int)))
# print(np.sum((benign_y != adv_y).astype(int)))

relevant_indices = (label[:,0] == benign_y) & (benign_y != adv_y)
adv_x = adv_x[relevant_indices]
adv_y = adv_y[relevant_indices]
benign_y = benign_y[relevant_indices]
benign_x = benign_x[relevant_indices]
label = label[relevant_indices]

print(adv_x.shape,benign_x.shape,benign_y.shape,label.shape,fgsm_set_y.shape)

np.savez('cifar100_adversary.npz', adv_x=adv_x, benign_y=label, benign_x=benign_x)