import os, sys, cv2, random, pathlib
import numpy as np
import threading
from keras.preprocessing.image import ImageDataGenerator
from normal_denormal_labeling import Normalize, DeNormalize

#IMAGE_SIZE = 256

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g



class MyDataGenerator(object):
    # imageのサイズ、フォルダ、ターゲットのフォルダ、バッチサイズを指定
    def __init__(self, size, img_dir, gt_dir, batch_size,equalize,color_dic):
        self.image_w = size[0]
        self.image_h = size[1]
        # 元画像の入っているフォルダ
        self.img_dir = img_dir
        # マスク画像の入っているフォルダ
        self.gt_dir = gt_dir
        self.batch_size=batch_size
        self.equalize = equalize
        # あらかじめインスタンス化しておく。
        self.norm = Normalize(color_dic)



    # 値を-1から1に正規化する関数
    @staticmethod
    def normalize_x(image):
        image = image / 127.5 - 1
        return image

    @staticmethod
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


    @staticmethod
    def list_shuffle(a,b,seed):
        np.random.seed(seed)
        l = list(zip(a, b))
        np.random.shuffle(l)
        a1, b1 = zip(*l)
        a2 = list(a1)
        b2 = list(b1)
        return a2,b2

    # カラーで読み込んで、(histgram平坦化)、リサイズして、－１～１の範囲に正規化
    def load_image_x(self,src):
        if( self.equalize ):
            src = MyDataGenerator.equalizeImg(src)
        src = cv2.resize(src, (self.image_w, self.image_h))
        return MyDataGenerator.normalize_x(src)

    # ラベル画像を読み込む関数
    # 元画像をw, h, color_dic + 1分にする。
    # 全てが0 or 0.8-1に
    def load_image_y(self,src):
        img = cv2.resize(src,(self.image_w, self.image_h))
        return self.norm.normalize(img)

    def fpath_making(self):
        seed = 1
        imgfpath_list = []
        maskfpath_list = []
        file_list = os.listdir(self.img_dir)
        for file in file_list:
            imgfpath = os.path.join(self.img_dir, file)
            maskfpath = os.path.join(self.gt_dir, file)
            imgfpath_list.append(imgfpath)
            maskfpath_list.append(maskfpath)
        imgfpath_list, maskfpath_list = MyDataGenerator.list_shuffle(imgfpath_list, maskfpath_list, seed)
        return imgfpath_list, maskfpath_list


    def data_gen(self,X_train,Y_train):
        # ImageDataGeneratorは設定した後、fitして使うもの
        # fitの引数はX_train
        # 代入しなくても、fitするだけでrandomにX_trainの中身をいじってくる。
        IDG = ImageDataGenerator(
            # rescale=1./255,
            rotation_range= 2,
            width_shift_range=0.01,
            height_shift_range=0.01,
            shear_range=0,
            zoom_range= 0,
            horizontal_flip=False,
            vertical_flip=False,
        )
        IDG.fit(X_train,augment = True, seed = 1)
        IDG.fit(Y_train,augment = True, seed = 1)
        return X_train,Y_train



    # imgfpath_list(元画像fileパスの一覧)
    # maskfpath_list(マスク画像fileパスの一覧)
    # のi番目～ｂ個分だけ読み取って、
    # 4次元テンソルにする。
    def mini_batch(self,imgfpath_list,maskfpath_list,i):
        X_train = []
        y_train = []
        for b in range(self.batch_size):
            imgfpath = imgfpath_list[i +b]
            maskfpath = maskfpath_list[i + b]
            img = cv2.imread(imgfpath)
            mask = cv2.imread(maskfpath)
            imageTmp = MyDataGenerator.load_image_x(self,img)
            X_train.append(imageTmp)
            MaskTmp = MyDataGenerator.load_image_y(self,mask)
            y_train.append(MaskTmp)
        X_train = np.array(X_train, dtype=np.float32)
        Y_train = np.array(y_train, dtype=np.float32)
        X_train1, Y_train1 = MyDataGenerator.data_gen(self,X_train,Y_train)
        return X_train1, Y_train1

    @threadsafe_generator
    def datagen(self,imgfpath_list,maskfpath_list): # data generator
        while True:
            for i in range(0, len(imgfpath_list) - self.batch_size, self.batch_size):
                x, t = MyDataGenerator.mini_batch(self,imgfpath_list,maskfpath_list,i)
                yield x, t

    # datagen_all
    # imgfpath_listとmaskfpath_listがそろっている前提
    # 元画像と新規画像を４次元ベクトルにして出力。
    def datagen_all(self, imgfpath_list,maskfpath_list):
        X_val = []
        y_val = []
        for idx in range(len(imgfpath_list)):
            fpath = imgfpath_list[idx]
            print(fpath)
            img = cv2.imread(fpath)
            maskfpath = maskfpath_list[idx]
            mask = cv2.imread(maskfpath)

            imageTmp = MyDataGenerator.load_image_x(self,img)
            maskTmp = MyDataGenerator.load_image_y(self,mask)
            X_val.append(imageTmp)
            y_val.append(maskTmp)
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)
        return X_val,y_val
