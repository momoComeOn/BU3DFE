#coding=gbk
import numpy as np
import math
import heapq
from functools import cmp_to_key
from curvature import get_curvature
from quadtree import quad_tree
import hdf5storage as hs
import h5py
import random

EPS = 1e-10
cur_fpath = None


def is_zero(x):
  return math.fabs(x) < EPS
  
  
def dist(a,b):
  return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
  
  
def set_cur_fpath(fpath):
  global cur_fpath
  cur_fpath = fpath

  
class winged_edge:
  def __init__(self, eid, v1, v2):
    self.eid = eid
    self.v = [v1, v2]
    self.f = []
    self.fe = []
    self.len = None
  def cal_len(self, vertices):
    self.len = dist(vertices[self.v[0]], vertices[self.v[1]])
  def add_face(self, fi):
    self.f.append(fi)
  def add_face_edge(self, e1, e2):
    self.fe.append([e1,e2])
  def find_opp_vid_on_face(self, faces, fi):
    fid = self.f[fi]
    opp_vid_on_face = -1
    for k in range(3):
      if faces[fid][k] != self.v[0] and faces[fid][k] != self.v[1]:
        if opp_vid_on_face == -1:
          opp_vid_on_face = k
        else:
          raise Exception('%s: Find more than one opposite vertices in face(%d)' % (cur_fpath, fid))
    return opp_vid_on_face

    
class bndry_vert:
  def __init__(self, vid):
    self.vid = vid
    self.next = None
    self.prev = None
    self.angle = None
    self.len = None
    self.nei_fid = None
    self.deleted = False
  def cal_info(self, vertices, faces, len, nei_fid):
    self.len = len
    self.nei_fid = nei_fid
    v0 = vertices[self.vid]
    v1 = vertices[self.next.vid]
    v2 = vertices[self.prev.vid]
    e1 = v1 - v0
    e2 = v2 - v0
    norm_e1_e2 = np.linalg.norm(e1) * np.linalg.norm(e2)
    if is_zero(norm_e1_e2):
      norm_e1_e2 += EPS
    cos_e1e2 = float(np.dot(e1, e2)) / (norm_e1_e2)
    if cos_e1e2 > 1:
      cos_e1e2 = 1
    elif cos_e1e2 < -1:
      cos_e1e2 = -1
    tmp_angle = math.acos(cos_e1e2)
    normvec1 = np.cross(e1, e2)
    v0 = vertices[faces[self.nei_fid][0]]
    v1 = vertices[faces[self.nei_fid][1]]
    v2 = vertices[faces[self.nei_fid][2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normvec2 = np.cross(e1, e2)
    same_direction = float(np.dot(normvec1, normvec2))
    if is_zero(tmp_angle-math.pi):
      self.angle = 3*math.pi
    elif same_direction > 0:
      self.angle = tmp_angle
    else:
      self.angle = 2*math.pi - tmp_angle
      
      
def read_mat(fn, faces_sub_1=False):
  data = h5py.File(fn, 'r')
  vertices = data['vertices'][:,:].T
  faces = data['faces'][:,:].T
  if faces_sub_1:
    faces= faces-1
  return vertices, faces
 

def read_obj(fn, faces_sub_1=True):
  f = open(fn, 'r')
  vertices = []
  faces = []
  for line in f:
    line = line.strip().split(' ')
    if line[0] == 'v':
      vertices.append([eval(line[1]), eval(line[2]), eval(line[3])])
    if line[0] == 'f':
      faces.append([eval(line[1]), eval(line[2]), eval(line[3])])
  f.close()
  vertices = np.array(vertices).astype(np.float32)
  faces = np.array(faces).astype(np.int32)
  if faces_sub_1:
    faces = faces-1
  return vertices, faces

  
def write_obj(obj_fn, vertices, faces):
  f = open(obj_fn, 'w')
  for i in range(vertices.shape[0]):
    f.write('v %f %f %f\n' % (vertices[i][0], vertices[i][1], vertices[i][2]))
  if faces is not None:
    faces = faces + 1
    for i in range(faces.shape[0]):
      f.write('f %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2]))
  f.close()


#只能算出法向量所在的直线，无法确定正负号
def write_pdir(pdir_fn, vertices, faces, save_normal_dir=True, mat_fn=''):
  vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
  H, K, k1, k2, shape_index, curvedness_index, pdir1, pdir2 = get_curvature(vertices, vert_nei_vert, nei_k=5)
  f = open(pdir_fn, 'w')
  for i in range(pdir1.shape[0]):
    f.write('%f %f %f\n' % (pdir1[i][0], pdir1[i][1], pdir1[i][2]))
    # f.write('%f %f %f\n' % (pdir2[i][0], pdir2[i][1], pdir2[i][2]))
    # if save_normal_dir:
      # normal_dir = np.cross(pdir1[i], pdir2[i])
      # f.write('%f %f %f\n' % (normal_dir[0], normal_dir[1], normal_dir[2]))
  f.close()
  mat_data = {}
  mat_data[u'shape_index'] = shape_index
  hs.writes(mat_data, mat_fn, options=hs.Options(matlab_compatible=True))
  
def write_normal(norm_fn, vertices, faces):
  _, vert_nei_face = construct_vert_nei(vertices, faces)
  v0 = vertices[faces[:,0]]
  v1 = vertices[faces[:,1]]
  v2 = vertices[faces[:,2]]
  e1 = v1 - v0
  e2 = v2 - v0
  face_normal_vec = np.cross(e1, e2)
  face_normal_vec /= np.linalg.norm(face_normal_vec, axis=1, keepdims=True)
  vert_normal_vec = []
  for i in range(len(vert_nei_face)):
    nv = np.zeros(3)
    for vnf in vert_nei_face[i]:
      nv += face_normal_vec[vnf, :]
    nv /= len(vert_nei_face[i])
    vert_normal_vec.append(nv)
  vert_normal_vec = np.array(vert_normal_vec)
  mat_data = {}
  mat_data[u'normals'] = vert_normal_vec
  hs.writes(mat_data, norm_fn, options=hs.Options(matlab_compatible=True))
  

#将0-1之间的数值转换成RGB或灰度值
def float2RGB(value, gray=False, zero_to_white=True):
  if value < 0:
    value = 0
  elif value > 1:
    value = 1
  
  if gray:
    return value,value,value
  if zero_to_white and value == 0:
    return 1,1,1

  if value < 1/6:     # 000-001 黑-蓝
    r = 0
    g = 0
    b = value*6
  elif value < 1/3:   # 001-011 蓝-青
    r = 0
    g = 6*value-1
    b = 1
  elif value < 1/2:   # 011-010 青-绿
    r = 0
    g = 1
    b = 3-6*value
  elif value < 2/3:   # 010-110 绿-黄
    r = 6*value-3
    g = 1
    b = 0
  elif value < 5/6:   # 110-100 黄-红
    r = 1
    g = 5-6*value
    b = 0
  elif value <= 1:    # 100-101 红-紫
    r = 1
    g = 0
    b = 6*value-5
  # else:               # 101-111 紫-白
    # r = 1
    # g = 6*value-6
    # b = 1
  return r, g, b

  
def write_wrl(wrl_fn, vertices, faces, color=None, gray=False, normalize='max_min', zero_to_white=True):
  f = open(wrl_fn, 'w')
  f.write('Shape {\n')
  f.write('\tappearance Appearance {\n')
  f.write('\t\tmaterial Material {\n')
  f.write('\t\t\tdiffuseColor 0.5882 0.5882 0.5882\n')
  f.write('\t\t\tambientIntensity 1.0\n')
  f.write('\t\t\tspecularColor 0 0 0\n')
  f.write('\t\t\tshininess 0.145\n')
  f.write('\t\t\ttransparency 0\n')
  f.write('\t\t}\n')
  f.write('\t}\n')
  f.write('\tgeometry IndexedFaceSet {\n')
  f.write('\t\tccw TRUE\n')
  f.write('\t\tsolid TRUE\n')
  f.write('\t\tcoord Coordinate { point [\n')
  # print point
  for i in range(vertices.shape[0]):
      f.write('\t\t\t%f,%f,%f,\n' % (vertices[i][0], vertices[i][1], vertices[i][2]))
  f.write('\t\t\t]\n')
  f.write('\t\t}\n')
  f.write('\t\tcoordIndex [\n')
  # print face
  for i in range(faces.shape[0]):
      f.write('\t\t\t%d,%d,%d,-1,\n' % (faces[i][0], faces[i][1], faces[i][2]))
  f.write('\t\t]\n')
  
  if color is not None:
    f.write('\t\tcolor Color { color [\n')
    # print color
    if normalize == 'max_min':
      max_color = np.max(color)
      min_color = np.min(color)
      color = (color-min_color)/(max_color-min_color)
    elif normalize == 'mean_var':
      color = ((color-np.mean(color))/(3*np.std(color))+1)/2
      reserve_color = color[(color>=0) & (color<=1)]
      max_color = np.max(reserve_color)
      min_color = np.min(reserve_color)
      color = (color-min_color)/(max_color-min_color)
    for i in range(vertices.shape[0]):
      r,g,b = float2RGB(color[i],gray,zero_to_white)
      f.write('\t\t\t%f,%f,%f,\n' % (r, g, b))
    f.write('\t\t\t]\n')
    f.write('\t\t}\n')
    f.write('\t\tcolorIndex   [\n')
    # print face
    for i in range(faces.shape[0]):
      f.write('\t\t\t%d,%d,%d,-1,\n' % (faces[i][0], faces[i][1], faces[i][2]))
    f.write('\t\t]\n')
    f.write('\t\tcreaseAngle 1.5\n')
    f.write('\t\tcolorPerVertex TRUE\n')

  f.write('\t}\n')
  f.write('}')

  f.close()

 
def cal_areas(vertices, faces):
  vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
  v0, v1, v2 = vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]
  face_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)/2
  areas = []
  for nei_face in vert_nei_face:
    areas.append([face_areas[nei_face].sum()/3])
  return np.array(areas).astype(np.float32)
  
  
def write_mat(mat_fn, vertices, faces, save_areas=True):
  mat_data = {}
  mat_data[u'vertices'] = vertices
  mat_data[u'faces'] = faces
  if save_areas:
    mat_data[u'areas'] = cal_areas(vertices, faces)
  hs.writes(mat_data, mat_fn, options=hs.Options(matlab_compatible=True))
  
  
def cmp(p1, p2):
  if is_zero(p1[0]-p2[0]):
    if is_zero(p1[1]-p2[1]):
      if is_zero(p1[2]-p2[2]):
        return 0
      elif p1[2] < p2[2]:
        return -1
      elif p1[2] > p2[2]:
        return 1
    elif p1[1] < p2[1]:
      return -1
    elif p1[1] > p2[1]:
      return 1
  elif p1[0] < p2[0]:
    return -1
  elif p1[0] > p2[0]:
    return 1


def delete_vertices(delete, vertices, faces, delete_large_part=False, del_faces=None):
  global cur_fpath
  del_num = delete.count(True)
  if not delete_large_part and del_num * 200 > vertices.shape[0]:
    print('Warning: Delete too many vertices(%d) in %s' % (del_num, cur_fpath))
  if del_num == 0 and (del_faces is None or del_faces.count(True) == 0):
    return vertices, faces
  ret_verts = []
  old_to_new = {}
  new_index = 0
  for i in range(vertices.shape[0]):
    if not delete[i]:
      ret_verts.append(vertices[i])
      old_to_new[i] = new_index
      new_index += 1
    else:
      old_to_new[i] = -1
  ret_faces = []
  for i in range(faces.shape[0]):
    if (del_faces is None) or not del_faces[i]:
      f0 = old_to_new[faces[i][0]]
      f1 = old_to_new[faces[i][1]]
      f2 = old_to_new[faces[i][2]]
      if f0 != -1 and f1 != -1 and f2 != -1:
        ret_faces.append([f0, f1, f2])
  ret_verts = np.array(ret_verts).astype(np.float32)
  ret_faces = np.array(ret_faces).astype(np.int32)
  return ret_verts, ret_faces

  
def can_be_deleted(vid, winged_edges, vert_nei_edge):
  for eid in vert_nei_edge[vid]:
    if len(winged_edges[eid].fe) > 2:
      return False
  return True
  

def dfs_edge(eid, winged_edges, vert_nei_edge, visited, del_verts, just_visit_2_face_edge=False):
  stack = [eid]
  while len(stack) != 0:
    cur_eid = stack.pop()
    visited[cur_eid] = True
    we = winged_edges[cur_eid]
    for i in range(2):
      if not del_verts[we.v[i]]:
        if not just_visit_2_face_edge:
          del_verts[we.v[i]] = True
        elif can_be_deleted(we.v[i], winged_edges, vert_nei_edge):
          del_verts[we.v[i]] = True
    for face_edge in we.fe:
      for fe in face_edge:
        if not visited[fe]:
          if not just_visit_2_face_edge:
            stack.append(fe)
          elif len(winged_edges[fe].fe) <= 2:
            stack.append(fe)
  
  
def construct_winged_edges(vertices, faces):
  winged_edges = []
  vert_nei_edge = [[] for i in range(vertices.shape[0])]
  edge_dict = {}
  edge_index = 0
  for i in range(faces.shape[0]):
    face_winged_edges = []
    for j in range(3):
      v1 = faces[i][j]
      v2 = faces[i][(j+1)%3]
      str1 = '%d_%d' % (v1, v2)
      str2 = '%d_%d' % (v2, v1)
      if str1 in edge_dict:
        eid = edge_dict[str1]
        we = winged_edges[eid]
      else:
        edge_dict[str1] = edge_index
        edge_dict[str2] = edge_index
        we = winged_edge(edge_index, v1, v2)
        winged_edges.append(we)
        edge_index += 1
      we.add_face(i)
      face_winged_edges.append(we)
      for k in range(2):
        if we.eid not in vert_nei_edge[we.v[k]]:
          vert_nei_edge[we.v[k]].append(we.eid)
    for j in range(3):
      face_winged_edges[j].add_face_edge(face_winged_edges[(j+1)%3].eid, face_winged_edges[(j+2)%3].eid)
  return winged_edges, vert_nei_edge

  
def construct_vert_nei(vertices, faces):
  vert_nei_vert = [[] for i in range(vertices.shape[0])]
  vert_nei_face = [[] for i in range(vertices.shape[0])]
  for fid in range(faces.shape[0]):
    for i in range(3):
      vid = faces[fid][i]
      for j in range(2):
        vid_j = faces[fid][(i+j+1)%3]
        if vid_j not in vert_nei_vert[vid]:
          vert_nei_vert[vid].append(vid_j)
      if fid not in vert_nei_face[vid]:
        vert_nei_face[vid].append(fid)
  return vert_nei_vert, vert_nei_face
  
  
def del_outlier_verts_faces(vertices, faces, args=None):
  winged_edges, vert_nei_edge = construct_winged_edges(vertices, faces)
  visited = [False for i in range(len(winged_edges))]
  max_del_num = -1
  max_del_ind = -1
  del_verts_list = []
  for eid in range(len(winged_edges)):
    if not visited[eid]:
      del_verts_candidate = [False for i in range(vertices.shape[0])]
      dfs_edge(eid, winged_edges, vert_nei_edge, visited, del_verts_candidate)
      del_verts_list.append(del_verts_candidate)
      del_num = del_verts_candidate.count(True)
      if del_num > max_del_num:
        max_del_num = del_num
        max_del_ind = len(del_verts_list)-1
  if len(del_verts_list) == 1 and del_verts_list[0].count(True) == vertices.shape[0]:
    if args is not None:
      args['del_verts'] = [False for i in range(vertices.shape[0])]
    return vertices, faces
  del_faces = [False for i in range(faces.shape[0])]
  for i,del_verts_cand in enumerate(del_verts_list):
    if i != max_del_ind:
      for j in range(faces.shape[0]):
        if del_verts_cand[faces[j][0]] == True and del_verts_cand[faces[j][1]] == True and del_verts_cand[faces[j][2]] == True:
          del_faces[j] = True
    else:
      del_verts = del_verts_cand
  for i in range(len(del_verts)):
    del_verts[i] = not del_verts[i]
  if args is not None:
    args['del_verts'] = del_verts
  vertices, faces = delete_vertices(del_verts, vertices, faces, del_faces=del_faces)
  return vertices, faces
  
  
def sort_verts_faces(vertices, faces):
  cp_verts = []
  for i in range(vertices.shape[0]):
    cp_verts.append(np.array([vertices[i][0], vertices[i][1], vertices[i][2], i]).astype(np.float32))
  key = cmp_to_key(cmp)
  cp_verts.sort(key=key)
  old_to_new = {}
  for i,v in enumerate(cp_verts):
    old_to_new[int(v[3])] = i
  
  cp_faces = []
  for i in range(faces.shape[0]):
    cp_faces.append(np.array([old_to_new[faces[i][0]], old_to_new[faces[i][1]], old_to_new[faces[i][2]]]).astype(np.int32))
  key = cmp_to_key(cmp)
  cp_faces.sort(key=key)
  
  vertices = []
  for v in cp_verts:
    vertices.append([v[0], v[1], v[2]])
  faces = []
  for f in cp_faces:
    faces.append([f[0], f[1], f[2]])
  vertices = np.array(vertices).astype(np.float32)
  faces = np.array(faces).astype(np.int32)
  return vertices, faces


def del_repeat_verts_faces(vertices, faces, already_sorted=False):
  global cur_fpath
  if not already_sorted:
    vertices,faces = sort_verts_faces(vertices, faces)
  last_v = None
  ret_verts = []
  old_to_new = {}
  new_index = 0
  for i in range(vertices.shape[0]):
    if last_v is None or not is_zero(dist(last_v,vertices[i])):
      ret_verts.append(vertices[i])
      last_v = vertices[i]
      old_to_new[i] = new_index
      new_index += 1
    else:
      old_to_new[i] = -1
  mapped_faces = []
  for i in range(faces.shape[0]):
    f0 = old_to_new[faces[i][0]]
    f1 = old_to_new[faces[i][1]]
    f2 = old_to_new[faces[i][2]]
    if f0 != -1 and f1 != -1 and f2 != -1:
      mapped_faces.append([f0, f1, f2])
  
  def is_diff_faces(f1, f2):
    for i in range(3):
      if f1[i] not in f2:
        return True
    return False
  
  last_f = None
  ret_faces = []
  for i in range(len(mapped_faces)):
    if last_f is None or is_diff_faces(last_f, mapped_faces[i]):
      ret_faces.append(mapped_faces[i])
      last_f = mapped_faces[i]
  
  ret_verts = np.array(ret_verts).astype(np.float32)
  ret_faces = np.array(ret_faces).astype(np.int32)
  return ret_verts, ret_faces
  
  
def del_protruding_faces(vertices, faces):
  winged_edges, vert_nei_edge = construct_winged_edges(vertices, faces)
  # Maybe something wrong with the block of code
  # global_del_verts = [False for i in range(vertices.shape[0])]
  # for we in winged_edges:
    # if len(we.fe) > 2:
      # max_del_verts_num = -1
      # max_del_index = -1
      # second_max_del_verts_num = -1
      # second_max_index = -1
      # del_verts_candidates = []
      # for ind,face_edge in enumerate(we.fe):
        # visited = [False for i in range(len(winged_edges))]
        # del_verts_candidate = [False for i in range(vertices.shape[0])]
        # dfs_edge(face_edge[0], winged_edges, vert_nei_edge, visited, del_verts_candidate, just_visit_2_face_edge=True)
        # del_verts_candidates.append(del_verts_candidate)
        # del_num = del_verts_candidate.count(True)
        # if del_num > max_del_verts_num:
          # second_max_del_verts_num = max_del_verts_num
          # second_max_index = max_del_index
          # max_del_verts_num = del_num
          # max_del_index = ind
        # elif del_num > second_max_del_verts_num:
          # second_max_del_verts_num = del_num
          # second_max_index = ind
      # for i,del_verts in enumerate(del_verts_candidates):
        # if i != max_del_index and i != second_max_index:
          # for i in range(len(global_del_verts)):
            # global_del_verts[i] = global_del_verts[i] or del_verts[i]
  # if global_del_verts.count(True) * 200 > vertices.shape[0]:
  global_del_verts = [False for i in range(vertices.shape[0])]
  no_change = True
  for we in winged_edges:
    if len(we.fe) > 2:
      global_del_verts[we.v[0]] = True
      global_del_verts[we.v[1]] = True
      no_change = False
  if no_change:
    return vertices, faces, no_change
  vertices, faces = delete_vertices(global_del_verts, vertices, faces)
  if global_del_verts.count(True) > 0:
    vertices, faces = del_outlier_verts_faces(vertices, faces)
  return vertices, faces, no_change

  
def find_bndry(bndry_entry, vert_nei_vert, vert_nei_face, faces, visited):
  global cur_fpath
  bevid = bndry_entry.vid
  next_vid = -1
  candidate_next = []
  for vi in vert_nei_vert[bevid]:
    if len(vert_nei_vert[vi]) != len(vert_nei_face[vi]):
      cnt = 0
      for fid in vert_nei_face[bevid]:
        if vi == faces[fid][0] or vi == faces[fid][1] or vi == faces[fid][2]:
          cnt += 1
      if cnt == 1:
        candidate_next.append(vi)
  if len(candidate_next) == 2:
    for cand in candidate_next:
      for fid in vert_nei_face[bevid]:
        tmp_list = ['%d_%d' % (faces[fid][0], faces[fid][1]),
                    '%d_%d' % (faces[fid][1], faces[fid][2]),
                    '%d_%d' % (faces[fid][2], faces[fid][0])]
        if ('%d_%d' % (cand, bevid)) in tmp_list:
          if next_vid == -1:
            next_vid = cand
          else:
            raise Exception('%s: More than two next vertices at first.' % cur_fpath, bevid)
  else:
    raise Exception('%s: More than two next vertices at first.' % cur_fpath, bevid)       
  if next_vid == -1:
    raise Exception('%s: Can\'t determine the boundary\'s direction.' % cur_fpath, bevid)    
  visited[bevid] = True
  bndry_length = 1
  bndry_entry.next = bndry_vert(next_vid)
  bndry_entry.next.prev = bndry_entry
  last_bndry = bndry_entry
  while True:
    cur_bndry = last_bndry.next
    visited[cur_bndry.vid] = True
    bndry_length += 1
    next_vid = -1
    candidate_next = []
    for nvid in vert_nei_vert[cur_bndry.vid]:
      if not visited[nvid] and len(vert_nei_vert[nvid]) != len(vert_nei_face[nvid]):
        candidate_next.append(nvid)
    if len(candidate_next) == 1:
      next_vid = candidate_next[0]
    elif len(candidate_next) > 1:
      cvid = cur_bndry.vid
      for cand_vid in candidate_next:
        cnt = 0
        for fid in vert_nei_face[cand_vid]:
          if cvid == faces[fid][0] or cvid == faces[fid][1] or cvid == faces[fid][2]:
            cnt += 1
        if cnt == 1:
          if next_vid == -1:
            next_vid = cand_vid
          else:
            raise Exception('%s: More than one next vertex.' % cur_fpath, cvid)
    if next_vid == -1:
      cur_bndry.next = bndry_entry
      bndry_entry.prev = cur_bndry
      break
    else:
      cur_bndry.next = bndry_vert(next_vid)
      cur_bndry.next.prev = cur_bndry
    last_bndry = cur_bndry
  return bndry_length

  
def cal_bndry_info(bndry_entry, vertices, faces, vert_nei_face):
  cur_bndry = bndry_entry
  bndry_heap = []
  bndry_dict = []
  total_len = 0
  total_num = 0
  while True:
    cvid = cur_bndry.vid
    nvid = cur_bndry.next.vid
    length = dist(vertices[cvid], vertices[nvid])
    for fid in vert_nei_face[cvid]:
      if nvid == faces[fid][0] or nvid == faces[fid][1] or nvid == faces[fid][2]:
        nei_fid = fid
        break
    cur_bndry.cal_info(vertices, faces, len=length, nei_fid=nei_fid)
    bndry_dict.append(cur_bndry)
    heapq.heappush(bndry_heap, (cur_bndry.angle, len(bndry_dict)-1))
    total_len += length
    total_num += 1
    if nvid == bndry_entry.vid:
      break
    cur_bndry = cur_bndry.next
  mean_len = total_len/total_num
  return bndry_dict, bndry_heap, mean_len, total_num
  
 
def linked_list_delete(bndry):
  next_bndry = bndry.next
  prev_bndry = bndry.prev
  next_bndry.prev = prev_bndry
  prev_bndry.next = next_bndry
  
  
def linked_list_replace(old_bndry, new_bndry):
  next_bndry = old_bndry.next
  prev_bndry = old_bndry.prev
  new_bndry.next = next_bndry
  new_bndry.prev = prev_bndry
  next_bndry.prev = new_bndry
  prev_bndry.next = new_bndry
 
  
def fill_hole(vertices, faces):
  vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
  visited = [False for i in range(vertices.shape[0])]
  bndrys = []
  max_len_bndry = -1
  max_len_index = -1
  for i in range(vertices.shape[0]):
    if not visited[i] and len(vert_nei_vert[i]) != len(vert_nei_face[i]):
      bndry_entry = bndry_vert(i)
      try:
        bndry_length = find_bndry(bndry_entry, vert_nei_vert, vert_nei_face, faces, visited)#首先确定方向
      except Exception as e:
        del_verts = [False for i in range(vertices.shape[0])]
        del_verts[e.args[1]] = True
        vertices, faces = delete_vertices(del_verts, vertices, faces)
        return fill_hole(vertices, faces)
      bndrys.append(bndry_entry)
      if bndry_length > max_len_bndry:
        max_len_bndry = bndry_length
        max_len_index = len(bndrys)-1
  if len(bndrys) > 1:
    for i,bndry_entry in enumerate(bndrys):
      if i != max_len_index:
        old_verts_num = vertices.shape[0]
        delete = [True for i in range(old_verts_num)]
        cur_bndry = bndry_entry
        while True:
          delete[cur_bndry.vid] = False
          cur_bndry = cur_bndry.next
          if cur_bndry.vid == bndry_entry.vid:
            break
        bndry_dict, bndry_heap, mean_len, bndry_length = cal_bndry_info(bndry_entry, vertices, faces, vert_nei_face)#这里假设各个孔洞相互独立
        while True:
          while True:
            min_angle_data = heapq.heappop(bndry_heap)
            cur_bndry = bndry_dict[min_angle_data[1]]
            if not cur_bndry.deleted:
              break
          next_bndry = cur_bndry.next
          prev_bndry = cur_bndry.prev
          if next_bndry.next.next.vid == cur_bndry.vid:
            new_face = np.array([cur_bndry.vid, next_bndry.vid, prev_bndry.vid]).astype(np.int32)
            faces = np.row_stack((faces, new_face))
            break
          new_len = dist(vertices[next_bndry.vid], vertices[prev_bndry.vid])
          if new_len < 2*mean_len:
            new_face = np.array([cur_bndry.vid, next_bndry.vid, prev_bndry.vid]).astype(np.int32)
            faces = np.row_stack((faces, new_face))
            new_next_bndry = bndry_vert(next_bndry.vid)
            new_prev_bndry = bndry_vert(prev_bndry.vid)
            linked_list_delete(cur_bndry)
            linked_list_replace(next_bndry, new_next_bndry)
            linked_list_replace(prev_bndry, new_prev_bndry)
            new_next_bndry.cal_info(vertices, faces, len=next_bndry.len, nei_fid=next_bndry.nei_fid)
            new_prev_bndry.cal_info(vertices, faces, len=new_len, nei_fid=faces.shape[0]-1)
            cur_bndry.deleted = True
            next_bndry.deleted = True
            bndry_dict.append(new_next_bndry)
            heapq.heappush(bndry_heap, (new_next_bndry.angle, len(bndry_dict)-1))
            prev_bndry.deleted = True
            bndry_dict.append(new_prev_bndry)
            heapq.heappush(bndry_heap, (new_prev_bndry.angle, len(bndry_dict)-1))
          else:
            vertices = np.row_stack((vertices, (vertices[next_bndry.vid] + vertices[prev_bndry.vid]) / 2))
            delete.append(False)
            new_vid = vertices.shape[0]-1
            new_faces = np.array([[cur_bndry.vid, next_bndry.vid, new_vid],
                                  [prev_bndry.vid, cur_bndry.vid, new_vid]]).astype(np.int32)
            faces = np.row_stack((faces, new_faces))
            new_cur_bndry = bndry_vert(new_vid)
            new_next_bndry = bndry_vert(next_bndry.vid)
            new_prev_bndry = bndry_vert(prev_bndry.vid)
            linked_list_replace(cur_bndry, new_cur_bndry)
            linked_list_replace(next_bndry, new_next_bndry)
            linked_list_replace(prev_bndry, new_prev_bndry)
            new_cur_bndry.cal_info(vertices, faces, len=new_len/2, nei_fid=faces.shape[0]-2)
            new_next_bndry.cal_info(vertices, faces, len=next_bndry.len, nei_fid=next_bndry.nei_fid)
            new_prev_bndry.cal_info(vertices, faces, len=new_len/2, nei_fid=faces.shape[0]-1)
            cur_bndry.deleted = True
            bndry_dict.append(new_cur_bndry)
            heapq.heappush(bndry_heap, (new_cur_bndry.angle, len(bndry_dict)-1))
            next_bndry.deleted = True
            bndry_dict.append(new_next_bndry)
            heapq.heappush(bndry_heap, (new_next_bndry.angle, len(bndry_dict)-1))
            prev_bndry.deleted = True
            bndry_dict.append(new_prev_bndry)
            heapq.heappush(bndry_heap, (new_prev_bndry.angle, len(bndry_dict)-1))
        sub_verts, sub_faces = delete_vertices(delete, vertices, faces, delete_large_part=True)
        sub_vert_nei_vert, sub_vert_nei_face = construct_vert_nei(sub_verts, sub_faces)
        L = np.zeros((sub_verts.shape[0], sub_verts.shape[0]))
        b = np.row_stack((sub_verts[:bndry_length], np.zeros((sub_verts.shape[0]-bndry_length, 3))))
        for i in range(L.shape[0]):
          L[i][i] = 1
          if i >= bndry_length:
            w = -1.0/len(sub_vert_nei_vert[i])
            for vid in sub_vert_nei_vert[i]:
              L[i][vid] = w
        new_sub_verts = np.dot(np.dot(np.linalg.inv(np.dot(L.T, L)),L.T),b)
        vertices = np.row_stack((vertices[:old_verts_num], new_sub_verts[bndry_length:]))
  return vertices, faces

  
def mean_filter(vertices, faces):
  vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
  has_outlier = False
  for i in range(vertices.shape[0]):
    vert = vertices[i]
    for nvid in vert_nei_vert[i]:
      vert = vert + vertices[nvid]
    if len(vert_nei_vert[i]) == 0:
      has_outlier = True
    vert = vert / (len(vert_nei_vert[i])+1)
    for j in range(3):
      vertices[i][j] = vert[j]
  if has_outlier:
    vertices, faces = del_outlier_verts_faces(vertices, faces)
  return vertices, faces
  

def del_protruding_faces_fill_hole(vertices, faces):
  vertices,faces,no_change = del_protruding_faces(vertices, faces)
  vertices,faces = fill_hole(vertices, faces)
  vertices,faces,no_change = del_protruding_faces(vertices, faces)
  cnt = 0
  while not no_change:
    vertices,faces = fill_hole(vertices, faces)
    vertices,faces,no_change = del_protruding_faces(vertices, faces)
    cnt += 1
    if cnt == 10:
      raise Exception('%s: Can\'t fix it' % fpath)
  return vertices, faces
  
  
def sub_remesh(vertices, faces, args):
  verts_num = args['verts_num']
  if vertices.shape[0] < verts_num:
    print('Call sub_remesh() but vertices\' number %d < %d in %s\nCall add_remesh() instead.' % (vertices.shape[0], verts_num, cur_fpath))
    return add_remesh(vertices, faces, args)
  vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
  try:
    H, K, k1, k2, shape_index, curvedness_index = get_curvature(vertices, vert_nei_vert, nei_k=5)
  except Exception as e:
    raise Exception('%s: %s' % (cur_fpath, e.args[0]))
  curv = np.fabs(H)+np.fabs(K)
  key = cmp_to_key(lambda i1,i2: curv[i1]-curv[i2])
  min_cur_indices = sorted(range(curv.shape[0]), key=key)
  vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
  sub_num = vertices.shape[0] - verts_num
  del_verts = [False for i in range(vertices.shape[0])]
  can_sub = [True for i in range(vertices.shape[0])]
  for i in min_cur_indices:
    if sub_num == 0:
      break
    if can_sub[i] and len(vert_nei_vert[i]) == len(vert_nei_face[i]):
      del_verts[i] = True
      can_sub[i] = False
      for j in vert_nei_vert[i]:
        can_sub[j] = False
        can_sub[j] = False
      sorted_nei_vids = []
      while True:
        if len(sorted_nei_vids) == len(vert_nei_vert[i]):
          break
        if len(sorted_nei_vids) == 0:
          fid = vert_nei_face[i][0]
          for k in range(3):
            if faces[fid][k] == i:
              break
          sorted_nei_vids.append(faces[fid][(k+1)%3])
          sorted_nei_vids.append(faces[fid][(k+2)%3])
          last_face = fid
        else:
          last_vid = sorted_nei_vids[-1]
          for j in vert_nei_face[last_vid]:
            if j != last_face:
              find_next = False
              for k in range(3):
                if faces[j][k] == i:
                  find_next = True
                  break
              if find_next:
                sorted_nei_vids.append(faces[j][(k+2)%3])
                break
          last_face = j
      new_faces = np.zeros((len(sorted_nei_vids)-2, 3)).astype(np.int32)
      for j in range(new_faces.shape[0]):
        new_faces[j][0] = sorted_nei_vids[0]
        new_faces[j][1] = sorted_nei_vids[j+1]
        new_faces[j][2] = sorted_nei_vids[j+2]
      faces = np.row_stack((faces, new_faces))
      sub_num -= 1
  vertices, faces = delete_vertices(del_verts, vertices, faces)
  if sub_num != 0:
    return sub_remesh(vertices, faces, args)
  return vertices, faces
  
  
def add_remesh(vertices, faces, args):
  verts_num = args['verts_num']
  if vertices.shape[0] > verts_num:
    print('Call add_remesh() but vertices\' number %d > %d in %s\nCall sub_remesh() instead.' % (vertices.shape[0], verts_num, cur_fpath))
    return sub_remesh(vertices, faces, args)
  winged_edges, vert_nei_edge = construct_winged_edges(vertices, faces)
  for we in winged_edges:
    we.cal_len(vertices)
  key = cmp_to_key(lambda i1,i2: winged_edges[i2].len-winged_edges[i1].len)
  sorted_indices = sorted(range(len(winged_edges)), key=key)
  add_num = verts_num - vertices.shape[0]
  del_faces = [False for i in range(faces.shape[0])]
  can_add = [True for i in range(len(winged_edges))]
  for i in sorted_indices:
    if add_num == 0:
      break
    if can_add[i]:
      we = winged_edges[i]
      if len(we.f) > 2 or len(we.f) < 1:
        raise Exception('%s: There are %d faces with this edge(%d,%d).' % (cur_fpath, len(we.f), we.v[0], we.v[1]))
      vertices = np.row_stack((vertices, (vertices[we.v[0]] + vertices[we.v[1]])/2))
      can_add[i] = False
      for j in range(len(we.f)):
        can_add[we.fe[j][0]] = False
        can_add[we.fe[j][1]] = False
        fid = we.f[j]
        del_faces[fid] = True
        opp_vid_on_face = we.find_opp_vid_on_face(faces, j)
        new_faces = np.array([[faces[fid][opp_vid_on_face], faces[fid][(opp_vid_on_face+1)%3], vertices.shape[0]-1],
                              [faces[fid][opp_vid_on_face], vertices.shape[0]-1, faces[fid][(opp_vid_on_face+2)%3]]]).astype(np.int32)
        faces = np.row_stack((faces, new_faces))
        del_faces.append(False)
        del_faces.append(False)
      add_num -= 1
  del_verts = [False for i in range(vertices.shape[0])]
  vertices, faces = delete_vertices(del_verts, vertices, faces, del_faces=del_faces)
  if add_num != 0:
    return add_remesh(vertices, faces, args)
  return vertices, faces
  
  
def remesh(vertices, faces, args):
  verts_num = args['verts_num']
  if vertices.shape[0] > verts_num:
    return sub_remesh(vertices, faces, args)
  elif vertices.shape[0] < verts_num:
    return add_remesh(vertices, faces, args)
  else:
    return vertices, faces
    
   
def sphere_normalize(vertices, faces):
  vertices = vertices - vertices.mean(axis=0)
  dist = np.sqrt(vertices[:,0]**2 + vertices[:,1]**2 + vertices[:,2]**2)
  vertices = vertices / np.max(dist)
  return vertices, faces
  
  
def point_is_occlusive(vp, v0, v1, v2):
  # e1 = v1 - v0
  # e2 = v2 - v0
  # det = e2[0]*e1[1] - e2[1]*e1[0]
  # if is_zero(det):
    # return False
  # ep = vp - v0
  # t1 = (e2[0]*ep[1] - ep[0]*e2[1]) / det
  # t2 = (e1[1]*ep[0] - ep[1]*e1[0]) / det
  # if t1 >= 0 and t2 >= 0 and t1 + t2 <= 1:
    # return True
  # else:
    # return False
  e1 = v1 - v0
  e2 = v2 - v0
  det = e2[:,0]*e1[:,1] - e2[:,1]*e1[:,0]
  vp = vp[np.newaxis,:]
  ep = vp - v0
  t1 = (e2[:,0]*ep[:,1] - ep[:,0]*e2[:,1]) / det
  t2 = (e1[:,1]*ep[:,0] - ep[:,1]*e1[:,0]) / det
  cond1 = np.logical_and(t1 >= 0, t2 >= 0)
  cond2 = np.logical_and(cond1, t1 + t2 <= 1)
  cond = np.logical_and(cond2, t1*e1[:,2] + t2*e2[:,2] + v0[:,2] > vp[:,2]+EPS)
  if cond.any():
    return True
  else:
    return False
  
  
def del_occlusion(vertices, faces, args=None):
  max_len = -np.inf
  v0_xy = vertices[faces[:,0]][:,:2]
  v1_xy = vertices[faces[:,1]][:,:2]
  v2_xy = vertices[faces[:,2]][:,:2]
  e0_xy = v1_xy - v0_xy
  e1_xy = v2_xy - v1_xy
  e2_xy = v0_xy - v2_xy
  e_max = [np.max(np.sum(e0_xy**2, axis=1)), 
           np.max(np.sum(e1_xy**2, axis=1)),
           np.max(np.sum(e2_xy**2, axis=1))]
  for em in e_max:
    if max_len < em:
      max_len = em
  max_len = math.sqrt(max_len)
  search_range = 1.01 * max_len
  
  min_split_dist = 0.5 * search_range
  qt = quad_tree(min_split_dist, vertices)
  
  del_verts = [False for i in range(vertices.shape[0])]
  _, vert_nei_face = construct_vert_nei(vertices, faces)

  for i in range(vertices.shape[0]):
    x = vertices[i][0]
    y = vertices[i][1]
    nei_pids = qt.find_points(x-search_range, x+search_range, y-search_range, y+search_range)
    fids = []
    for npid in nei_pids:
      for fid in vert_nei_face[npid]:
        if fid not in vert_nei_face[i]:
          fids.append(fid)
    fids = np.array(fids).astype(np.int32)
    if point_is_occlusive(vertices[i], vertices[faces[fids,0]], vertices[faces[fids,1]], vertices[faces[fids,2]]):
      del_verts[i] = True
  if args is not None:
    args['del_verts'] = del_verts
  return delete_vertices(del_verts, vertices, faces, delete_large_part=True)
  

def rotation_angle(vertices, faces, args):
  xa = args['xa']
  ya = args['ya']
  za = args['za']
  # if xa is None:
    # xa = (random.random()-0.5) * math.pi * 0.9
  # if ya is None:
    # ya = (random.random()-0.5) * math.pi * 0.9
  # if za is None:
    # za = (random.random()-0.5) * math.pi * 0.9
  RX = np.array([[1, 0, 0],
                 [0, math.cos(xa), -math.sin(xa)],
                 [0, math.sin(xa), math.cos(xa)]]).astype(np.float32)
  RY = np.array([[math.cos(ya), 0, math.sin(ya)],
                 [0, 1, 0],
                 [-math.sin(ya), 0, math.cos(ya)]]).astype(np.float32)
  RZ = np.array([[math.cos(za), -math.sin(za), 0],
                 [math.sin(za), math.cos(za), 0],
                 [0, 0, 1]]).astype(np.float32)
  vertices = np.dot(vertices, RX)
  vertices = np.dot(vertices, RY)
  vertices = np.dot(vertices, RZ)
  return vertices, faces