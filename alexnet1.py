# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import NPYS_Modify

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X, Y = NPYS_Modify.load_data()


# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 48, 9, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = tflearn.batch_normalization(network)
#network = local_response_normalization(network)
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = tflearn.batch_normalization(network)
#network = local_response_normalization(network)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = tflearn.batch_normalization(network)
#network = local_response_normalization(network)
network = fully_connected(network, 4096)
network = tflearn.batch_normalization(network)
network = dropout(network, 0.5)
network = tflearn.activation(network,activation='tanh')
network = fully_connected(network, 2048)
network = dropout(network, 0.5)
network = tflearn.activation(network,activation='tanh')
network = fully_connected(network, 50, activation='softmax')#50 means the number of speakers
network = regression(network, optimizer='Adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.load('./saved_model/augment_model.tflearn')
model.fit(X, Y, n_epoch=30, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=256, snapshot_step=10,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
model.save('./saved_model/augment_model.tflearn')
