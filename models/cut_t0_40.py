import sys

sys.path.append("../Kfbreader")

import Kfbreader.kfbReader as kr
import cv2
import numpy as np
from tqdm import *
import os

treshold = 200000
def filtercut(img):

    dst3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)
    a = np.sum(dst3==0)
    return a

def get_filelist(dir,save_path):

    for home, dirs, files in os.walk(dir):
        count = 0
        for filename in tqdm(files):
                count = count + 1
                if count == 100:
                  break
                kfb2jpg(home,filename,save_path)

def kfb2jpg(home,filename,save_path):
    file_prefix = filename.split('.')[0]
    size = 1000
    reader = kr.reader()
    scale = 40
    print(os.path.join(home,file_prefix + '.kfb'))
    kr.reader.ReadInfo(reader, os.path.join(home,file_prefix + '.kfb'), scale, False)
    # readScale = reader.getReadScale()
    # print('readScale',readScale)
    reader.setReadScale(scale)
    # print('readScale',reader.getReadScale())
    width,Height = reader.getWidth(),reader.getHeight()
    num_h,num_w = Height // size,width // size
    for i in trange(num_w):
        # imgtemp = reader.ReadRoi(i * size, 0, size, Height, scale)
        for j in range(num_h):
            img = reader.ReadRoi(i*size,j*size,size,size,scale)
            # img = imgtemp[j * size:(j + 1) * size, 0:1000]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if filtercut(gray) > treshold and img.shape[0]==1000 and img.shape[1] == 1000:#切图
                save_name = filename + '_' + 'annoname' + '_total_' + str(i * j) + ".jpg"
                cv2.imwrite(os.path.join(save_path, save_name), img)

if __name__ == '__main__':
    folder = "../dataset/kfb"
    save_path = '../dataset/tis'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    get_filelist(folder,save_path)
