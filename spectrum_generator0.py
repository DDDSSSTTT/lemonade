# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:46:07 2019

@author: D S T
"""
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
#生成一个需要产生语谱数组的文件列表
filepath = "./new_data_set/simple_test_set/" 
jpgpath = "./new_data_set/simple_test_set/spectro/"
#添加路径 \\是\的转义写法，用在windows下
filenamelist= os.listdir(filepath) #得到文件夹下的所有文件名称  

# 窗长20ms， 窗移时窗长的0.5倍cd
for filename in filenamelist:
    if filename.endswith(".wav"):
        # 调用wave模块中的open函数，打开语音文件。
        f = wave.open(filepath+filename,'rb')
        # 得到语音参数
        params = f.getparams()
        nchannels, sampwidth, framerate,nframes = params[:4]
        # 得到的数据是字符串，需要将其转成int型
        strData = f.readframes(nframes)
        wavaData = np.fromstring(strData,dtype=np.int16)
        # 归一化
        wavaData = wavaData * 1.0/max(abs(wavaData))
        # .T 表示转置
        wavaData = np.reshape(wavaData,[nframes,nchannels]).T
        f.close()
        # 绘制频谱
        plt.specgram(wavaData[0],Fs = framerate,scale_by_freq=True,sides='default')
        plt.ylabel('Frequency')
        plt.xlabel('Time(s)')
        #plt.show()
        plt.axis('off')

        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        fig = plt.gcf()
        if "."in filename:
            Filename = filename.split(".")[0]
        plt.savefig(jpgpath + Filename+'.jpg', bbox_inches = 'tight', dpi=200, pad_inches = 0)
        plt.close()  
        #new_spectrum=fetch.normalizeSpectrum(speech_spectrum,height=227,length=227)
        #numpy.save(npypath+filename.strip(".wav")+".npy",new_spectrum)
#load_spectrum=numpy.load(npypath+filenamelist[15].strip(".wav")+".npy")
#plt.imshow(load_spectrum)