import numpy as np

EPS = 1e-6

def rect_intersction(x_bg_n, x_ed_n, y_bg_n, y_ed_n, x_bg_s, x_ed_s, y_bg_s, y_ed_s):
  intersec = False
  inner = False
  if x_bg_n >= x_bg_s and x_ed_n <= x_ed_s and y_bg_n >= y_bg_s and y_ed_n <= y_ed_s:
    intersec = True
    inner = True
    return intersec, inner#, x_bg_n, x_ed_n, y_bg_n, y_ed_n
  else:
    rect_x_min = max(x_bg_n, x_bg_s)
    rect_x_max = min(x_ed_n, x_ed_s)
    if rect_x_min > rect_x_max:
      return intersec, inner#, None, None, None, None
      
    rect_y_min = max(y_bg_n, y_bg_s)
    rect_y_max = min(y_ed_n, y_ed_s)
    if rect_y_min > rect_y_max:
      return intersec, inner#, None, None, None, None
      
    intersec = True
    return intersec, inner#, rect_x_min, rect_x_max, rect_y_min, rect_y_max

class quad_node:
  def __init__(self, x_bg, x_ed, y_bg, y_ed, pids=None):
    self.sub_nodes = None
    self.x_bg, self.x_ed = x_bg, x_ed
    self.y_bg, self.y_ed = y_bg, y_ed
    if pids is None:
      self.pids = []
    else:
      self.pids = pids
  def add_pid(self, pid):
    self.pids.append(pid)

    
class quad_tree:
  def __init__(self, min_split_dist, vertices):
    self.min_split_dist = min_split_dist
    self.vertices = vertices.tolist()
    maxs = np.max(vertices, axis=0)
    mins = np.min(vertices, axis=0)
    x_max, y_max = maxs[0], maxs[1]
    x_min, y_min = mins[0], mins[1]
    self.root = quad_node(x_min-EPS, x_max+EPS, y_min-EPS, y_max+EPS, [i for i in range(len(self.vertices))])
    self.build_tree(self.root)
  
  def build_tree(self, node):
    if node is None or \
       len(node.pids) == 1 or \
       (node.x_ed - node.x_bg < self.min_split_dist and node.y_ed - node.y_bg < self.min_split_dist):
      return
    if node.sub_nodes is None:
      node.sub_nodes = [None]*4
    x_mid = (node.x_bg + node.x_ed) / 2
    y_mid = (node.y_bg + node.y_ed) / 2
    sub_x_range = [[node.x_bg, x_mid], [x_mid, node.x_ed]]
    sub_y_range = [[node.y_bg, y_mid], [y_mid, node.y_ed]]
    for vi in node.pids:
      v = self.vertices[vi]
      x, y = v[0], v[1]
      snid = -1
      if x < x_mid:
        if y < y_mid:
          snid = 0
        else:
          snid = 1
      else:
        if y < y_mid:
          snid = 2
        else:
          snid = 3
      if node.sub_nodes[snid] is None:
        node.sub_nodes[snid] = quad_node(sub_x_range[snid//2][0], sub_x_range[snid//2][1], sub_y_range[snid%2][0], sub_y_range[snid%2][1])
      node.sub_nodes[snid].add_pid(vi)
    for s_node in node.sub_nodes:
      self.build_tree(s_node)
      
  def find_points_recursion(self, node, x_bg, x_ed, y_bg, y_ed):
    if node is None:
      return []
    intersec, inner = rect_intersction(node.x_bg, node.x_ed, node.y_bg, node.y_ed, x_bg, x_ed, y_bg, y_ed)
    if not intersec:
      return []
    if inner:
      return node.pids
    ret = []
    if node.sub_nodes is None:
      for pid in node.pids:
        x, y = self.vertices[pid][0], self.vertices[pid][1]
        if x >= x_bg and x <= x_ed and y >= y_bg and y <= y_ed:
          ret.append(pid)
      return ret
    else:
      for s_node in node.sub_nodes:
        ret += self.find_points_recursion(s_node, x_bg, x_ed, y_bg, y_ed)
      return ret
  
  def find_points(self, x_bg, x_ed, y_bg, y_ed):
    return self.find_points_recursion(self.root, x_bg, x_ed, y_bg, y_ed)