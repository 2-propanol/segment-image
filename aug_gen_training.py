import os, sys
import cv2
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# ファイルのディレクトリをパスに含める(自前実装関数includeのため)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_augment_png import data_augment_copy
from PSPNet import PSPNet50
from dataGenUNet import MyDataGenerator
from training import train_unet
from utils_unet import folder_create, folder_delete, model_compile_unet, plot_hist_dice

# data_augment_png
# 元の画像ファイルをpng形式でデータ拡張or copyする。
# data_augmentが拡張、data_copyはコピー

# model_list_unetがのModelsがモデルを定義している。UNetのmodel

# dataGenUNetはUnet用のdataを読み取る。
#


if __name__ == '__main__':
    trainroot1 = "train"
    trainroot2 = "train2"
    folder_create(trainroot2)
    valroot = "val"
    model_file = "model.hdf5"
    BATCH_SIZE = 4
    NUM_EPOCH = 20
    # sizeおよびch
    size = [512,512]
    input_ch = 3

    # 1は黒目、２は白目
    color_dic = {1:[255,255,255]}
    output_ch = len(color_dic) + 1
    # データ拡張
    data_augment_copy(trainroot1, trainroot2)

    # modelのcompile
    input_shape = [size[0], size[1], input_ch]
    model = PSPNet50(input_shape = input_shape, n_labels = output_ch)
    # model.compile(loss=, optimizer="adadelta", metrics=["accuracy"])
    model_compile_unet(model)
    # model構築（Trainとvalidationで）
    history = train_unet(trainroot2,valroot,model_file,model,size, BATCH_SIZE, NUM_EPOCH, color_dic)
    plot_hist_dice(history)
    # trainroot2の消去
    folder_delete(trainroot2)
