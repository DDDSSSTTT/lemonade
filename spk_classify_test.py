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


speakers=[]
#speakers = data.get_speakers()
file=open('spkrs_list.txt','r')  
lines=file.readlines();
for line in lines:
    speaker=line.replace('\n', '')
    speakers.append(speaker)
number_classes=len(speakers)
print("speakers",speakers)

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

try:model.load('./saved_model/1_model.tflearn')
finally:ts_path="./new_data_set/simple_test_set/"
v_counter=0
samples=fetch.random_sample(ts_path,1)
for sample in samples:
    wave, sr = librosa.load(ts_path+sample, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    
    mfcc=np.pad(mfcc,((0,0),(0,640-len(mfcc[0]))), mode='constant', constant_values=0)
    #demo=data.load_wav_file(ts_path + sample)
    demo=mfcc
    result=model.predict([demo])
    result=data.one_hot_to_item(result,speakers)
    validity=fetch.check_speaker(result,sample)
    print("predicted speaker for %s : result = %s validity = %d"%(sample,result,validity)) 
    # ~ 97% correct
    if validity:
        v_counter+=1
print(v_counter/len(samples))
