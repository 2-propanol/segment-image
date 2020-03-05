import os, sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# ファイルのディレクトリをパスに含める(自前実装関数includeのため)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataGenUNet import MyDataGenerator

# U-Netのトレーニングを実行する関数
def train_unet(trainroot,valroot,model_file,model,size, BATCH_SIZE, NUM_EPOCH, color_dic):
    # 訓練用データのfileパス作成
    trainImgfol = os.path.join(trainroot, "img")
    trainMaskfol = os.path.join(trainroot, "mask")
    myDataGenTrain = MyDataGenerator(size, trainImgfol, trainMaskfol, BATCH_SIZE,True, color_dic)
    imgfpath_list_train, maskfpath_list_train = myDataGenTrain.fpath_making()

    # 検証用データのfileパス作成
    valImgfol = os.path.join(valroot, "img")
    valMaskfol = os.path.join(valroot, "mask")
    myDataGenVal = MyDataGenerator(size, valImgfol, valMaskfol, BATCH_SIZE,True, color_dic)
    imgfpath_list_val, maskfpath_list_val = myDataGenVal.fpath_making()
    X_val, y_val = myDataGenVal.datagen_all(imgfpath_list_val,maskfpath_list_val)

    # 学習実行
    # val_lossが最小になったときのみmodelを保存
    mc_cb = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # 学習が停滞したとき、学習率を0.2倍に
    rl_cb = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, mode='auto',
                              epsilon=0.0001, cooldown=0, min_lr=0)
    # 学習が進まなくなったら、強制的に学習終了
    es_cb = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')

    history = model.fit_generator(myDataGenTrain.datagen(imgfpath_list_train,maskfpath_list_train),
        int(len(imgfpath_list_train) / BATCH_SIZE),
        validation_data=(X_val,y_val),
        callbacks=[mc_cb,rl_cb,es_cb],
        epochs=NUM_EPOCH,
        verbose=1)
    return history
