# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:24:53 2019

@author: yubin
"""
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import io

from PIL import Image

filepath = "./new_data_set/simple_test_set/spectro/" 
npypath = "./new_data_set/simple_test_set/npys/" 

AUGMENTATION=1#选择是否增强

filenamelist= os.listdir(filepath) #得到文件夹下的所有文件名称  

# 窗长20ms， 窗移时窗长的0.5倍cd
for filename in filenamelist:
    if filename.endswith(".jpg"):
        img = cv2.imread(filepath+filename, -1)

        f=1
        if AUGMENTATION==1:
                weight=len(img[0])
                height=len(img)
                G=4#可以改2
                for index in np.linspace(0,G-1,G):
                    #G等分
                    son_image = img[0:height,int(index*weight/G):int((1+index)*weight/G)]
                    
                    image = cv2.resize(son_image, (227, 227), interpolation=cv2.INTER_AREA)
                    numpy.save(npypath+filename.strip(".jpg")+"_"+str(int(index))+".npy", image)
                image=cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
                for index in np.linspace(G,2*G-1,G):
                    numpy.save(npypath+filename.strip(".jpg")+"_"+str(int(index))+".npy", image)
        else:
                image = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
                numpy.save(npypath+filename.strip(".jpg")+".npy", image)