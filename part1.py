#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import median_filter
from scipy import ndimage
from efficientnet.tfkeras import EfficientNetB4
import cv2
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle


# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils # we need this

    
"""
## Plots an adversarial perturbation, i.e., original input orig_x, adversarial example adv_x, and the difference (perturbation)
"""
def plot_adversarial_example(pred_fn, orig_x, adv_x, labels, fname='adv_exp.png', show=True, save=True):
    perturb = adv_x - orig_x
    
    # compute confidence
    in_label, in_conf = utils.pred_label_and_conf(pred_fn, orig_x)
    
    # compute confidence
    adv_label, adv_conf = utils.pred_label_and_conf(pred_fn, adv_x)
    
    titles = ['{} (conf: {:.2f})'.format(labels[in_label], in_conf), 'Perturbation',
              '{} (conf: {:.2f})'.format(labels[adv_label], adv_conf)]
    
    images = np.r_[orig_x, perturb, adv_x]
    
    # plot images
    utils.plot_images(images, fig_size=(8,3), titles=titles, titles_fontsize=12,  out_fp=fname, save=save, show=show)  


######### Prediction Fns #########

"""
## Basic prediction function
"""
def basic_predict(model, x):
    return model(x)


#### TODO: implement your defense(s) as a new prediction function
### Put your code here
# def median_filter(img, kernel_size):
#     filtered_img = np.zeros_like(img)
#     for i in range(img.shape[-1]):
#         filtered_img[..., i] = median_filter(img[..., i], size=kernel_size)
#     return filtered_img

# def defense_predict(model, x):
#     # apply a median filter to smooth the image
    
#     # x = medfilt2d(x, kernel_size=(3,3))
#     x = ndimage.median_filter(x, size=3) 
#     # apply a Gaussian blur to reduce the noise
    
#     x = gaussian_filter(x, sigma=2)
    
#     # x = x/255.0

#     return model(x)

def apply_filters(img):
    # apply a median filter to smooth the image
    img = median_filter(img, size=3)
    
    # apply a total variation filter to reduce noise and preserve edges
    img = denoise_tv_chambolle(img, weight=0.1)

    return img

def defense_predict(model, x):
    # apply filters to the input image
    x = apply_filters(x)
    
    # normalize the input image to a range of 0 to 1
    x = x / 255.0
    
    # pass the filtered and normalized image to the model for prediction
    return model.predict(x)




######### Membership Inference Attacks (MIAs) #########

"""
## A very simple threshold-based MIA
"""
def simple_conf_threshold_mia(predict_fn, x, thresh=0.9999):   
    pred_y = predict_fn(x)
    pred_y_conf = np.max(pred_y, axis=-1)
    return (pred_y_conf > thresh).astype(int)
    
    
#### TODO [optional] implement new MIA attacks.
#### Put your code here
  
  
######### Adversarial Examples #########

  
#### TODO [optional] implement new adversarial examples attacks.
#### Put your code here  
#### Note: you can have your code save the data to file so it can be loaded and evaluated in Main() (see below).
    
   
######### Main() #########
   
if __name__ == "__main__":


    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Scikit-learn version: ' + sklearn.__version__)
    print('### Tensorflow version: ' + tf.__version__)
    print('### TF Keras version: ' + keras.__version__)
    print('------------')


    # global parameters to control behavior of the pre-processing, ML, analysis, etc.
    seed = 42

    # deterministic seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    # train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
    train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data_cnn()
    # train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data_svhn()
    num_classes = len(labels)
    assert num_classes == 100 # cifar10
    
    ### load the target model (the one we want to protect)
    # target_model_fp = './cifar100_cnn_l1.h5'
    # target_model_fp = './cifar100_cnn_1.h5'
    # target_model_fp = './cifar100_new.h5'
    # target_model_fp = './Pre-Trained Model.h5'
    target_model_fp = './cifar100_model.h5'

    model, _ = utils.load_model(target_model_fp)
    print(model.summary())
    ## model.summary() ## you can uncomment this to check the model architecture (ResNet)
    
    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and test data
    train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print('[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
    
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense
    # predict_fn = lambda x: basic_predict(model, x)
    predict_fn = lambda x: defense_predict(model, x) 
    ### now let's evaluate the model with this prediction function
    pred_y = predict_fn(train_x)
    train_acc = np.mean(np.argmax(train_y, axis=-1) == np.argmax(pred_y, axis=-1))
    
    pred_y = predict_fn(test_x)
    test_acc = np.mean(np.argmax(test_y, axis=-1) == np.argmax(pred_y, axis=-1))
    print('[Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
        
    
    ### evaluating the privacy of the model wrt membership inference
    mia_eval_size = 2000
    mia_eval_data_x = np.r_[train_x[0:mia_eval_size], test_x[0:mia_eval_size]]
    mia_eval_data_in_out = np.r_[np.ones((mia_eval_size,1)), np.zeros((mia_eval_size,1))]
    assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple MIA Attack', simple_conf_threshold_mia))
    
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_data_x).reshape(-1,1)
        assert in_out_preds.shape == mia_eval_data_in_out.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_data_in_out, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        print(tn, fp, fn, tp)
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_adv = tp / (tp + fn) - fp / (fp + tn)
        attack_precision = tp / (tp + fp)
        attack_recall = tp / (tp + fn)
        attack_f1 = tp / (tp + 0.5*(fp + fn))
        print('{} --- Attack accuracy: {:.2f}%; advantage: {:.3f}; precision: {:.3f}; recall: {:.3f}; f1: {:.3f}'.format(attack_str, attack_acc*100, attack_adv, attack_precision, attack_recall, attack_f1))
    
    
    
    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []
    advexp_fps.append(('Adversarial examples attack0', 'advexp0.npz'))
    advexp_fps.append(('Adversarial examples attack1', 'advexp1.npz'))
    
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp = tup
        
        data = np.load(attack_fp)
        adv_x = data['adv_x']
        benign_x = data['benign_x']
        benign_y = data['benign_y']
        
        benign_pred_y = predict_fn(benign_x)
        #print(benign_y[0:10], benign_pred_y[0:10])
        benign_acc = np.mean(benign_y == np.argmax(benign_pred_y, axis=-1))
        
        adv_pred_y = predict_fn(adv_x)
        #print(benign_y[0:10], adv_pred_y[0:10])
        adv_acc = np.mean(benign_y == np.argmax(adv_pred_y, axis=-1))
        
        print('{} --- Benign accuracy: {:.2f}%; adversarial accuracy: {:.2f}%'.format(attack_str, 100*benign_acc, 100*adv_acc))
        
    print('------------\n')

    et = time.time()
    
    print('Elapsed time -- total: {:.1f} seconds (data & model loading: {:.1f} seconds)'.format(et - st, st_after_model - st))

    sys.exit(0)
