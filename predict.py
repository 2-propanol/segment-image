import os, sys
import cv2
import numpy as np
# ファイルのディレクトリをパスに含める(自前実装関数includeのため)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from keras.models import load_model
from PSPNet import PSPNet50
from normal_denormal_labeling import DeNormalize
from utils_unet import folder_create, folder_delete, model_compile_unet, plot_hist_dice, tversky_loss

# data_augment_png
# 元の画像ファイルをpng形式でデータ拡張or copyする。
# data_augmentが拡張、data_copyはコピー

# model_list_unetがのModelsがモデルを定義している。UNetのmodel

# dataGenUNetはUnet用のdataを読み取る。
#
# 値を-1から1に正規化する関数
def normalize_x(image):
    image = image / 127.5 - 1
    return image


def equalizeImg(img):
    # チャンネル分割
    img_bgr = cv2.split(img)
    # チャンネルごとに平均化処理
    for i in range(3):
        equ = cv2.equalizeHist(img_bgr[i])
        img_bgr[i] = equ
    # チャンネルマージ
    x = cv2.merge((img_bgr[0], img_bgr[1], img_bgr[2]))
    return x

def load_image_x(src,size):
    src = equalizeImg(src)
    src = cv2.resize(src, (size[0], size[1]))
    return normalize_x(src)


if __name__ == '__main__':
    test_img_folder = os.path.join("val","img")
    test_mask_folder = os.path.join("val", "mask2")
    folder_create(test_mask_folder)
    test_output_folder = os.path.join("val", "output")
    folder_create(test_output_folder)
    model_file = "model.hdf5"
    BATCH_SIZE = 5
    # sizeおよびch
    size = [512,512]
    input_ch = 3
    color_dic = {1:[255,255,255]}
    output_ch = len(color_dic) + 1
    input_shape = [size[0], size[1], input_ch]
    model = load_model(model_file,custom_objects={"tversky_loss":tversky_loss})
    model_compile_unet(model)

    # model構築（Trainとvalidationで）
    denorm = DeNormalize(color_dic)
    alpha = 0.7

    file_list = os.listdir(test_img_folder)
    for file in file_list:
        fpath = os.path.join(test_img_folder, file)
        src = cv2.imread(fpath)
        img = load_image_x(src,[512,512])
        x = np.array([img])
        predict = model.predict(x)
        tag = predict[0]
        mask = denorm.denormalize(tag)
        newfpath1 = os.path.join(test_mask_folder, file)
        newfpath2 = os.path.join(test_output_folder, file)
        cv2.imwrite(newfpath1, mask)
        mask2 = cv2.resize(mask,(src.shape[1],src.shape[0]))
        dst1 = np.uint8(alpha * src + (1 - alpha) * mask2)
        cv2.imwrite(newfpath2, dst1)
