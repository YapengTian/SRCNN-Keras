# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 10:17:31 2016

@author: Yapeng
"""

from __future__ import print_function
import numpy as np
from keras.layers import Input, Convolution2D, merge
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import h5py
import math

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 20.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

	##paramaters
batch_size = 128
nb_epoch = 50
#input imaage dimensions
img_rows, img_cols = 33, 33
out_rows, out_cols = 33, 33
#filter number
n1 = 64
n2 = 32
n3 = 1
#filter size
f1 = 9
f2 = 1
f3 = 5
##load train data
file = h5py.File('train/train_mscale.h5', 'r')
in_train = file['data'][:]
out_train = file['label'][:]
# .........
file.close()
#load validation data
file = h5py.File('train/test_mscale.h5', 'r')
in_test = file['data'][:]
out_test = file['label'][:]
# .........
file.close()
#convert data form 
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')
in_test = in_test.astype('float32')
out_test = out_test.astype('float32')
if K.image_dim_ordering() == 'th':
    in_train = in_train.reshape(in_train.shape[0], 1, img_rows, img_cols)
    in_test  = in_test.reshape(in_test.shape[0], 1, img_rows, img_cols)
    out_train = out_train.reshape(out_train.shape[0], 1, out_rows, out_cols)
    out_test = out_test.reshape(out_test.shape[0], 1, out_rows, out_cols)
    input_shape = (1, img_rows, img_cols)

#print number of training patches
print('in_train shape:', in_train.shape)
print(in_train.shape[0], 'train samples')
print(in_test.shape[0], 'test samples')
#SR Model
#input tensor for a 1_channel image region
x = Input(shape = input_shape)
c1 = Convolution2D(n1, f1,f1, activation = 'relu', init = 'he_normal', border_mode='same')(x)
c2 = Convolution2D(n2, f2, f2, activation = 'relu', init = 'he_normal', border_mode='same')(c1)
c3 = Convolution2D(n3, f3, f3, init = 'he_normal', border_mode='same')(c2)
model = Model(input = x, output = c3)
##compile
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
model.compile(loss='mse', metrics=[PSNRLoss], optimizer=adam)     
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
history = model.fit(in_train, out_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks = [lrate],
          verbose=1, validation_data=(in_test, out_test))            
print(history.history.keys())
#save model and weights
json_string = model.to_json()  
open('convert/srcnn_model.json','w').write(json_string)  
model.save_weights('convert/srcnn_model_weights.h5') 
# summarize history for loss
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('model loss')
plt.ylabel('PSNR/dB')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()




