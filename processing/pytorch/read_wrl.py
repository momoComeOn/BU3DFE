# -*-coding:utf-8-*-
import numpy as np

# path = u'E://数据库//BU-3DFE//BU-3DFE//F0001//F0001_AN01WH_F3D.wrl'


# def read_wrl(path):
#     '''
#     读取wrl文件中的point和coordIndex
#     '''
#     points = []
#     point_flag = False
#     coordindex_flag = False
#     coordindex = []
#     with open(path, 'r') as wrl:
#         contexts = wrl.readlines()
#         for con in contexts:
#             if point_flag:
#                 if ']' in con:
#                     point_flag = False
#                 else:
#                     points.append(con)
#             if 'point [' in con:
#                 point_flag = True
#             if coordindex_flag:
#                 if ']' in con:
#                     break
#                 coordindex.append(con)
#             if 'coordIndex [' in con:
#                 coordindex_flag = True
#     return points, coordindex


# def del_kong(points, style):
#     '''
#     删除元素中的空格以及句尾的，str转int和float
#     '''
#     result = []
#     for point in points:
#         point_ele = point.strip().split(' ')
#         if ',' in point_ele[-1]:
#             point_ele[-1] = point_ele[-1][:-1]
#         if style == 'float':
#             point_ele = [float(i) for i in point_ele[:3]]
#         if style == 'int':
#             point_ele = [int(i) for i in point_ele[:3]]
#         result.append(point_ele[:3])
#     return result


# points, coordindex = read_wrl(path)
# points = del_kong(points, 'float')
# coordindex = del_kong(coordindex, 'int')

# vertices = np.array(points)
# faces = np.array(coordindex)

import scipy.io as sio
aa = sio.loadmat('reason.mat')
vertices = aa['str']['ver'][0][0]
faces = aa['str']['face'][0][0] - 1

print 'size of vertices:'
print vertices.shape
print vertices
print 'size of faces:'
print faces.shape
print faces

from curvature import construct_vert_nei, get_curvature
from mesh_tools import del_outlier_verts_faces

# vertices, faces = del_outlier_verts_faces(vertices,faces)

print 'size of vertices:'
print vertices.shape
print vertices
print 'size of faces:'
print faces.shape
print faces

vert_nei_vert,ver_nei_face = construct_vert_nei(vertices, faces)
tmp = get_curvature(vertices,vert_nei_vert,nei_k=5)




sio.savemat('result1.mat',{'vertices':vertices,'shape_index':tmp[4]})

