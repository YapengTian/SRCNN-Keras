# SRCNN-Keras
Implement SRCNN via Keras with Theano as backend. For a fair comparison with published works, low-resolution images are produced by Matlab imresize function.

# Use pre-trained model
run SRCNN_test.m in “test” folder

# Training
1. generate training patches using matlab 
2. use Keras with Theano as backend to train SRCNN model
3. convert Keras model to .Mat for testing using Matconvnet

# How to train your model?
1. generate training patches 
2. run SRCNN.py to produce SRCNN model
3. run load_save.py first then save_model.m to produce Matconvnet model 

# Differences with the original implementation
1. use Adam to optimize the network for fast convergence
2. train *2,*3 and *4 simultaneously like the VDSR (CVPR 16)
 
# Dependencies
[Theano](http://github.com/Theano/Theano), [Keras](http://github.com/fchollet/keras), [Matconvnet](http://www.vlfeat.org/matconvnet/).


If this code is helpful for you, please cite this paper: "Image Super-Resolution Using Deep Convolutional Networks".



