# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:48:17 2019

@author: D S T
"""


import librosa
import numpy as np
import tflearn
import speech_data as data
import file_fetch_ts as fetch
import tensorflow as tf
import micro_live
tf.logging.set_verbosity(tf.logging.ERROR)

print("Loading speakers...")
speakers=[]
#speakers = data.get_speakers()
file=open('spkrs_list.txt','r')  
lines=file.readlines();
for line in lines:
    speaker=line.replace('\n', '')
    speakers.append(speaker)
number_classes=len(speakers)
print("speakers",speakers)
print("Initializing Networks...")
# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

input_layer = tflearn.input_data(shape=[20, 640]) #Two wave chunks, the time dimension of the wav file has been modified
fc1 = tflearn.fully_connected(input_layer, 128,name='fc1')
bn1 = tflearn.batch_normalization(fc1,name='bn1')
dp1 = tflearn.dropout(bn1, 0.5,name='dp1')
ac1 = tflearn.activation(bn1,activation='softmax',name='ac1')
#net = tflearn.fully_connected(net, 400, activation='softmax')
#net = tflearn.fully_connected(net, 200, activation='softmax')

fc2 = tflearn.fully_connected(ac1, 64,name='fc2')
bn2 = tflearn.batch_normalization(fc2,name='bn2')
dp2 = tflearn.dropout(bn2, 0.5,name='dp2')
ac2 = tflearn.activation(bn2,activation='softmax',name='ac2')
fc3 = tflearn.fully_connected(ac2, number_classes, activation='softmax',name='fc3')
rg1 = tflearn.regression(fc3, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(rg1)

model.load('./saved_model/1_model.tflearn')

print("Ready!")
while True:
    command=input("Enter your cmd: ")
    if command=="mic":
        r = micro_live.recoder()
        r.recoder()
        r.savewav("test.wav") 

        wave, sr = librosa.load('test.wav', mono=True)
        mfcc = librosa.feature.mfcc(wave, sr)
    
        mfcc=np.pad(mfcc,((0,0),(0,640-len(mfcc[0]))), mode='constant', constant_values=0)
        #demo=data.load_wav_file(ts_path + sample)
        demo=mfcc
        result=model.predict([demo])
        result=data.one_hot_to_item(result,speakers)
        #validity=fetch.check_speaker(result,'test.wav')
        print("predicted speaker for %s : result = %s"%('test.wav',result)) 