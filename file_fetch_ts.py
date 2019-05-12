#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:05:15 2019
@author:milan_boy
@author: dst
"""


import os, random,math
pathDir=[]
def random_sample(t_path,rate):
    for file in os.listdir(t_path):
        if file.endswith(".wav") or file.endswith(".npy"):
            pathDir.append(file)   
    
    
    filenumber=len(pathDir)

    picknumber=int(filenumber*rate) 
#按照rate比例从文件夹中取一定数量图片 
    samples = random.sample(pathDir, picknumber)
#print (sample) 
    return samples

def check_speaker(result,file,position):
    valid=0
    answer=extract(file,position)
    if answer==result:
        valid=1
    else:
        valid=0
    return valid

def extract(name,position):
    if "_" in name:    
        return name.split("_")[position]#the value "-2" can be changed
    else:
        return "0"

def get_spkrs(dir_path):
    spkrs=[]
    for file in os.listdir(dir_path):
        if file.endswith(".wav") or file.endswith(".npy"):
            if(extract(file)):
                name=extract(file)
                if name not in spkrs:
                    spkrs.append(extract(file))
    return spkrs
def one_hot_from_item(item, items):
	# items=set(items) # assure uniqueness
	x=[0]*len(items)# numpy.zeros(len(items))
	i=items.index(item)
	x[i]=1
	return x
import wave,numpy

def getSpectrum(filename, window_length_ms, window_shift_times):  
    # 读音频文件
    wav_file = wave.open(filename, 'r')
    # 获取音频文件的各种参数
    params = wav_file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    # 获取音频文件内的数据，不知道为啥获取到的竟然是个字符串，还需要在numpy中转换成short类型的数据
    str_data = wav_file.readframes(wav_length)
    wave_data = numpy.fromstring(str_data, dtype=numpy.int8)
    # 将窗长从毫秒转换为点数
    window_length = framerate * window_length_ms / 1000
    window_shift = int(window_length * window_shift_times)
    # 计算总帧数，并创建一个空矩阵
    nframe = (wav_length - (window_length - window_shift)) / window_shift
    spec = numpy.zeros((int(window_length/2), int(nframe)))
    # 循环计算每一个窗内的fft值
    for i in range(int(nframe)):
        start = int(i * window_shift)
        end = int(start + window_length)
        # [:window_length/2]是指只留下前一半的fft分量
        spec[:, i] = numpy.log(numpy.abs(numpy.fft.fft(wave_data[start:end]))+0.001)[:int(window_length/2)]
    return spec
def normalizeSpectrum(speech_spectrum,height,length):
        
    mat_h_floor=int((height-speech_spectrum.shape[0])/2)
    mat_h_ceil=math.ceil((height-speech_spectrum.shape[0])/2)
    mat_l_floor=int((length-speech_spectrum.shape[1])/2)
    mat_l_ceil=math.ceil((length-speech_spectrum.shape[1])/2)
    if mat_l_floor<0:
        speech_spectrum=speech_spectrum[:length]
        mat_l_floor=0
        mat_l_ceil=math.ceil(0)
    new_spectrum=numpy.pad(speech_spectrum, ((mat_h_floor, mat_h_ceil), (mat_l_floor, mat_l_ceil)), 'constant',constant_values=(0.01, 0.01))
    return new_spectrum
