#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import shutil
from keras.optimizers import Adam
# ファイルのディレクトリをパスに含める(自前実装関数includeのため)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras import backend as K

def folder_create(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return

def folder_delete(folder):
    shutil.rmtree(folder)
    return

# ダイス係数を計算する関数
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

# ロス関数
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def tversky_loss(y_true, y_pred):
    alpha = 0.7
    beta  = 0.3
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def model_compile_unet(model):
    model.compile(loss=tversky_loss, optimizer=Adam(lr = 0.001))
    return model

def plot_hist_dice(history):
    # historyオブジェクト（model.fit_generatorから生まれる）の属性として
    # .history["acc"]や.history["val_acc"]がある。
    # 元々のmovie名からタイトルを決定する
    history_file = "history.jpg"
    title = "loss_vs_Epoch"
    # axパターン
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.plot(history.history["loss"])
    ax.plot(history.history['val_loss'])
    ax.legend(['train', "val"], loc='upper left')  # 凡例ON
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.savefig(history_file)
    plt.clf()
    return
