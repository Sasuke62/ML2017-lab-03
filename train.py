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
img_list_face = get_image_dir('./datasets/original/face')
num_img_face = len(img_list_face)
img_face = []
img_face_matrix = []
for i in range(num_img_face):
    img_face.append(Image.open(img_list_face[i]).convert('L'))
    img_face[i].thumbnail((24, 24))
    img_face_matrix.append(np.array(img_face[i]))
'''
使用NPDFeature类的方法提取特征
使用pickle库中的dump()函数将预处理后的特征数据保存到缓存中
使用load()函数读取特征数据
'''
#x, y = np.shape(img_face_matrix[1])
#print(x,y)
img_face_feature = []
for i in range(10):
    print(i)
    img_face_feature.append(NPDFeature(img_face_matrix[i]))
    img_face_feature[i].extract()
#print(img_face_feature[0].features)

data_file = open('./data.pkl', 'wb')
for i in range(10):
    pickle.dump(img_face_feature, data_file)

data_file.close()

data_file = open('./data.pkl', 'rb')

feature = pickle.load(data_file)
print(feature)
#NPDFeature.extract()
#print(NPDFeature.features)

'''
测试输出图片
'''
#img_face[0].show()
#img_face[1].show()


#img = Image.open('./datasets/original/face/face_000.jpg')

#if __name__ == "__main__":
    # write your code here

#    pass

