
# coding: utf-8

# In[5]:

import os
import cv2
import numpy as np


# In[57]:

# mask1は含む
# mask2は含まれる
def convert_to_color_mask(folder1, folder2, folder3):
    if not os.path.isdir(folder3):
        os.mkdir(folder3)
    file_list = os.listdir(folder1)
    for file in file_list:
        fpath1 = os.path.join(folder1, file)
        fpath2 = os.path.join(folder2, file)
        fpath3 = os.path.join(folder3, file)
        th = 125
        # img1が大きい
        # img2が小さい
        img1 = cv2.imread(fpath1, 0)
        img11 = img1.copy()
        img11[img1 > th] = 255
        img11[img1 <= th] = 0
        img11 = np.array(img11 / 255, dtype = np.uint8)
        h,w = img11.shape[:2]
        if os.path.isfile(fpath2):
            img2 = cv2.imread(fpath2, 0)
            img2 = cv2.resize(img2,(w,h))
            img21 = img2.copy()
            img21[img2 > th] = 255
            img21[img2 <= th] = 0
            img21 = np.array(img21 / 255, dtype = np.uint8)
        else:
            img21 = np.zeros((h,w),dtype = np.uint8)
        img = np.zeros((h,w,3),dtype = np.uint8)
        img[:,:,0][(img11 == 1) & (img21 == 1)] = 255
        img[:,:,2][(img11 == 1) & (img21 == 0)] = 255
        cv2.imwrite(fpath3,img)


# In[27]:




# In[ ]:



