# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:24:53 2019

@author: yubin
"""
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import cv2


filepath = "./new_data_set/simple_test_set/augmentation/jpg/" 
npypath = "./new_data_set/simple_test_set/augmentation/NPYS/" 

filenamelist= os.listdir(filepath) #得到文件夹下的所有文件名称  

# 窗长20ms， 窗移时窗长的0.5倍cd
for filename in filenamelist:
    if filename.endswith(".jpg"):
        img = cv2.imread(filepath+filename, -1)
        image = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
        numpy.save(npypath+filename.strip(".jpg")+".npy", image)
        

