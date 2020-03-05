
# coding: utf-8

# In[24]:

import cv2
import os
import numpy as np


# In[25]:

class Normalize():
    def __init__(self,color_dic):
        self.color_dic = color_dic

    def color_pick(self,img,color):
        ch1 = np.zeros((img.shape[0],img.shape[1]))
        ch1[(img[:,:,0] == color[0])&(img[:,:,1] == color[1])&(img[:,:,2] == color[2])] = 1
        return ch1

    def normalize(self,img):
    # labelingの箱をつくる
        outputs = np.zeros((img.shape[0],img.shape[1],len(self.color_dic) + 1))
        # label1 - の中に、条件を満たすピクセルのみ1に変換していく
        for i in range(1,len(self.color_dic)+1):
            ch1 = self.color_pick(img,self.color_dic[i])
            outputs[:,:,i] = ch1
        # 1番上の層（つまり、背景以外が0のもの）のみ抽出
        ch1sum = np.sum(outputs[:,:,1:] , axis = 2)
        # 一番上の層に代入する板を用意
        ch0 = np.zeros((img.shape[0],img.shape[1]))
        # 背景以外が0のもののみを1に変換する
        ch0[ch1sum == 0] = 1
        # 1番上の層の中身を代入する。
        outputs[:,:,0] = ch0
        # 0.8-1の値にバラす
        return outputs


# In[28]:

class DeNormalize():
    def __init__(self,color_dic):
        self.color_dic = color_dic

    def denormalize(self, tag):
        tag_arr = np.argmax(tag,axis = 2)
        img = np.zeros((tag.shape[0],tag.shape[1],3),dtype = np.uint8)
        for i in range(len(self.color_dic)+1):
            if i == 0:
                for idx in range(3):
                    img[:,:,idx][tag_arr == i] = 0
            else:
                for idx in range(3):
                    img[:,:,idx][tag_arr == i] = self.color_dic[i][idx]
        return img
