#!/usr/bin/env python
#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os
import librosa
import numpy as np
import tflearn
import speech_data as data
import file_fetch_ts as fetch
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


speakers=[]
#speakers = data.get_speakers()
file=open('spkrs_list.txt','r')  
lines=file.readlines();
for line in lines:
    speaker=line.replace('\n', '')
    speakers.append(speaker)
number_classes=len(speakers)
print("speakers",speakers)

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
statistic_array=np.zeros((1,number_classes))
try:model.load('./saved_model/augment_model.tflearn')
finally:ts_path="./new_data_set/simple_test_set/npys/"
v_counter=0
samples=fetch.random_sample(ts_path,1)
for sample in samples:
    load_spectrum=np.load(ts_path+sample)
    #demo=np.reshape(load_spectrum,(227,227,1))
    demo=np.array(load_spectrum,dtype=np.float32)
    result1=model.predict([demo])
    result=data.one_hot_to_item(result1,speakers)
    validity=fetch.check_speaker(result,sample,-3)#-2or-3
    print("predicted speaker for %s : result = %s validity = %d"%(sample,result,validity)) 
    # ~ 97% correct
    if validity:
        v_counter+=1
    else:
        statistic_array=statistic_array+data.one_hot_from_item(fetch.extract(sample,-3),speakers)        
print(v_counter/len(samples))
