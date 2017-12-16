import numpy as np
import random
import matplotlib.pylab as plt
import pickle
import os

from feature import NPDFeature
from PIL import Image

def get_image_dir(dir_path):
    '''
    返回目录中所有jpg图像的文件名的列表
    '''
    return [os.path.join(dir_path, img) for img in os.listdir(dir_path)]

'''
加载原始图像并将图像resize成24*24并且转化为灰度图
'''
img_list = get_image_dir('./datasets/original/face')
num_img_face = len(img_list)
img_face = []

for i in range(num_img_face):
    img_face.append(Image.open(img_list[i]).convert('L'))
    img_face[i].thumbnail((24, 24))


img_face[0].show()
img_face[1].show()


#img = Image.open('./datasets/original/face/face_000.jpg')

#if __name__ == "__main__":
    # write your code here

#    pass

