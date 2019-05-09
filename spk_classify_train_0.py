#!/usr/bin/env python
#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import os
import librosa
import numpy as np
import tflearn
import speech_data as data
import file_fetch_ts as fetch

# Simple speaker recognition demo, with 99% accuracy in under a minute ( on digits sample )

# | Adam | epoch: 030 | loss: 0.05330 - acc: 0.9966 -- iter: 0000/1000
# 'predicted speaker for 9_Vicki_260 : result = ', 'Vicki'
import tensorflow as tf
print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
#if tf.__version__ >= '0.12' and os.name == 'nt':
#	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
#	quit() # why? works on Mac?

speakers = data.get_speakers()

number_classes=len(speakers)
#print("speakers",speakers)


file=open('spkrs_list.txt','w')  
for line in speakers:
    file.write(line+'\n')
file.close() 

batch=data.mfcc_batch_generator(batch_size=2000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
X,Y=next(batch)


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
try:model.fit(X, Y, n_epoch=500, show_metric=True)
finally:model.save('./saved_model/1_model.tflearn')

