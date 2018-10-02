# coding:utf-8
import os
from PIL import Image
import numpy as np
import shutil

def conver_SI():
    root = u'E:/数据库/BU3DFE-2D/BU3DFESI_from_SUNJIA'
    des = u'E:/数据库/BU3DFE-2D/BU3DFE-SI'

    if not os.path.exists(des):
        os.makedirs(des)

    files = os.listdir(root)

    for file in files:
        img = Image.open(os.path.join(root, file))

        out = img.transpose(Image.FLIP_TOP_BOTTOM)
        out = out.convert('L')
        out.save(os.path.join(des, file[:-7])+'.bmp')


def convert_2D():
    root = u'E:/数据库/BU-3DFE/BU-3DFE'
    des = u'E:/数据库/BU3DFE-2D/BU3DFE-2'

    files = os.listdir(root)
    for file in files:
        root_ = os.path.join(root,file)
        if os.path.isdir(root_):
            lists = os.listdir(root_)
            for l in lists:
                if '2D.bmp' in l:
                    shutil.copyfile(os.path.join(root_,l),os.path.join(des,l))


def make_lists():
    des = u'E:/数据库/BU3DFE-2D/BU3DFE-2'
    map = {'AN':'0','DI':'1','FE':'2','HA':'3','SA':'4','SU':'5'}
    files = os.listdir(des)
    with open('BU3DFE.txt','w') as txt:
        for file in files:
            if not 'NE' in file:
                txt.write(file+' '+map[file[6:8]]+'\n')


conver_SI()
