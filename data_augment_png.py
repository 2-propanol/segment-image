#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# usage: ./increase_picture.py hogehoge.jpg
#

import cv2
import numpy as np
import shutil
import sys
import os
from utils_unet import folder_create

class Trans:
    def __init__(self):
        # ルックアップテーブルの生成
        self.min_table = 50
        self.max_table = 205
        self.diff_table = self.max_table - self.min_table
        self.gamma1 = 0.75
        self.gamma2 = 1.5
        # 平滑化用
        self.average_square = (10,10)

        self.LUT_HC = np.arange(256, dtype = 'uint8' )
        self.LUT_LC = np.arange(256, dtype = 'uint8' )
        self.LUT_G1 = np.arange(256, dtype = 'uint8' )
        self.LUT_G2 = np.arange(256, dtype = 'uint8' )
        self.LUTs = []



        # ハイコントラストLUT作成
        for i in range(0, self.min_table):
            self.LUT_HC[i] = 0

        for i in range(self.min_table, self.max_table):
            self.LUT_HC[i] = 255 * (i - self.min_table) / self.diff_table

        for i in range(self.max_table, 255):
            self.LUT_HC[i] = 255

        # その他LUT作成
        for i in range(256):
            self.LUT_LC[i] = self.min_table + i * (self.diff_table) / 255
            self.LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / self.gamma1)
            self.LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / self.gamma2)

        self.LUTs.append(self.LUT_HC)
        self.LUTs.append(self.LUT_LC)
        self.LUTs.append(self.LUT_G1)
        self.LUTs.append(self.LUT_G2)

        self.func_list = [self.nothing_func, self.hc1_func, self.lc1_func, self.gamma1_func, self.gamma2_func,
                     self.blur_func, self.equalizeHistRGB_func, self.addGaussianNoise_func, self.addSaltPepperNoise_func]

    # ① 何もしない
    def nothing_func(self, src):
        return src

    # ② high contrast
    def hc1_func(self, src):
        img_dst = cv2.LUT(src, self.LUTs[0])
        return img_dst
    # ③ low_contrast1
    def lc1_func(self, src):
        img_dst = cv2.LUT(src, self.LUTs[1])
        return img_dst
    # ④ gamma補正1
    def gamma1_func(self,src):
        img_dst = cv2.LUT(src, self.LUTs[2])
        return img_dst
    # ⑤ gamma補正2
    def gamma2_func(self,src):
        img_dst = cv2.LUT(src, self.LUTs[3])
        return img_dst
    # ⑥　平滑化
    def blur_func(self,src):
        img_dst = cv2.blur(src, self.average_square)
        return img_dst
    # ⑦ヒストグラム均一化
    def equalizeHistRGB_func(self,src):

        RGB = cv2.split(src)
        Blue   = RGB[0]
        Green = RGB[1]
        Red    = RGB[2]
        for i in range(3):
            cv2.equalizeHist(RGB[i])

        img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
        return img_hist
    # ⑧ガウシアンノイズ
    def addGaussianNoise_func(self, src):
        row,col,ch= src.shape
        mean = 0
        var = 0.1
        sigma = 15
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = src + gauss

        return noisy
    # ⑨salt&pepperノイズ
    def addSaltPepperNoise_func(self,src):
        row,col,ch = src.shape
        s_vs_p = 0.5
        amount = 0.004
        out = src.copy()
        # Salt mode
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i-1 , int(num_salt))
                     for i in src.shape]
        out[coords[:-1]] = (255,255,255)

        # Pepper mode
        num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper))
                 for i in src.shape]
        out[coords[:-1]] = (0,0,0)
        return out
    # ⑩反対
    def lr_flip_func(self,src):
        img_dst = cv2.flip(src,1)
        return img_dst

    # ↑が関数の具体的定義
    # ①そのまま、 nothing
    # LUTsに含まれる
    # ②High contrast   high_contrast1
    # ③Low contrast    low_contrast1
    # ④Gamma変換1,     gamma1
    # ⑤Gamma変換2      gamma2
    # ⑥平滑化          blur
    # ⑦ヒストグラム均一化    equalizeHistRGB
    # ⑧ガウシアンノイズ      addGaussianNoise
    # ⑨ソルトペッパーノイズ  addSaltPepperNoise
    # 下が関数のリスト

    def convert(self, src, num):
        img_dst = self.func_list[num](src)
        return img_dst

# newfolderは保存先のフォルダ
# fileは元々のfile名（拡張子なし）→のちにjpgをたす
# srcは元の画像（numpy array型式）
# num_listはどの変換を適応するか
# modeはflipのどれを含めるか（2018/4/1時点では左右flipのみ）

# data_augmentはfileに様々な拡張を加えて出力。
def data_augment(newfolder,filewithext, src0):
    data = Trans()
    file = filewithext.split(".")[0]
    ext = filewithext.split(".")[1]
    for num in range(9):
        src1 = data.lr_flip_func(src0)
        dst0 = data.convert(src0,num)
        dst1 = data.convert(src1,num)
        newfile0 = "0" + "_" + str(num) + "_" + file + "." + ext
        newfile1 = "1" + "_" + str(num) + "_" + file + "." + ext
        newfpath0 = os.path.join(newfolder,newfile0)
        newfpath1 = os.path.join(newfolder,newfile1)
        cv2.imwrite(newfpath0, dst0)
        cv2.imwrite(newfpath1, dst1)

def data_copy(newfolder,filewithext, src0):
    data = Trans()
    file = filewithext.split(".")[0]
    ext = filewithext.split(".")[1]
    for num in range(9):
        src1 = data.lr_flip_func(src0)
        newfile0 = "0" + "_" + str(num) + "_" + file + "." + ext
        newfile1 = "1" + "_" + str(num) + "_" + file + "." + ext
        newfpath0 = os.path.join(newfolder,newfile0)
        newfpath1 = os.path.join(newfolder,newfile1)
        cv2.imwrite(newfpath0, src0)
        cv2.imwrite(newfpath1, src1)


def data_augment_copy(trainroot1, trainroot2):
    imgfolder = "img"
    maskfolder = "mask"
    train_img_folder1 = os.path.join(trainroot1, imgfolder)
    train_img_folder2 = os.path.join(trainroot2, imgfolder)
    folder_create(train_img_folder2)
    train_mask_folder1 = os.path.join(trainroot1, maskfolder)
    train_mask_folder2 = os.path.join(trainroot2, maskfolder)
    folder_create(train_mask_folder2)

    img_flist = os.listdir(train_img_folder1)
    print("img augment")
    for img_f in img_flist:
        img_fpath = os.path.join(train_img_folder1,img_f)
        img0 = cv2.imread(img_fpath)
        data_augment(train_img_folder2,img_f,img0)
    print("mask augment")
    mask_flist = os.listdir(train_mask_folder1)
    for mask_f in mask_flist:
        mask_fpath = os.path.join(train_mask_folder1,mask_f)
        mask0 = cv2.imread(mask_fpath)
        data_copy(train_mask_folder2,mask_f,mask0)

