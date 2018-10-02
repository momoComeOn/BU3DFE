# encoding:utf-8
import numpy as np
import math

# Using demo
# H 平均曲率，K 高斯曲率，k1 最大主曲率，k2 最小主曲率，pdir1 最大主曲率方向，pdir2 最小主曲率方向，
# shape_index 归一化曲率（0,1），curvedness_index 曲度

# vert_nei_vert, vert_nei_face = construct_vert_nei(vertices, faces)
# H, K, k1, k2, shape_index, curvedness_index, pdir1, pdir2 = get_curvature(vertices, vert_nei_vert, nei_k=5)


def construct_vert_nei(vertices, faces):
    vert_nei_vert = [[] for i in range(vertices.shape[0])]
    vert_nei_face = [[] for i in range(vertices.shape[0])]
    for fid in range(faces.shape[0]):
        for i in range(3):
            vid = faces[fid][i]
            for j in range(2):
                vid_j = faces[fid][(i+j+1) % 3]
                if vid_j not in vert_nei_vert[vid]:
                    vert_nei_vert[vid].append(vid_j)
            if fid not in vert_nei_face[vid]:
                vert_nei_face[vid].append(fid)
    return vert_nei_vert, vert_nei_face


def get_normal(norm_fn, vertices, faces):
    _, vert_nei_face = construct_vert_nei(vertices, faces)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
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
    return vert_normal_vec


def get_nei_k(vert_nei_vert, nei_k):
    nei_ring = [[{i}] for i in range(len(vert_nei_vert))]
    ret_nei = []
    for i in range(len(vert_nei_vert)):
        for k in range(nei_k):
            cur_ring = set({})
            last_ring = nei_ring[i][k]
            if k > 0:
                last_last_ring = nei_ring[i][k-1]
            else:
                last_last_ring = set({})
            for vid in last_ring:
                for vid_nei in vert_nei_vert[vid]:
                    if vid_nei not in last_last_ring and vid_nei not in last_ring and vid_nei not in cur_ring:
                        cur_ring.add(vid_nei)
            nei_ring[i].append(cur_ring)
        nei_i = set({})
        for ring_set in nei_ring[i]:
            nei_i = nei_i | ring_set
        ret_nei.append(nei_i)
    return ret_nei, nei_ring


def fit_quad_surface(vertices, vids):
    if len(vids) < 3:
        raise Exception('Too few vertices to fit the surface.')
    x = 0
    y = 0
    z = 0
    xy = 0
    xz = 0
    yz = 0
    xyz = 0
    x2 = 0
    y2 = 0
    x2y = 0
    x2z = 0
    y2z = 0
    xy2 = 0
    x3 = 0
    y3 = 0
    x4 = 0
    y4 = 0
    x2y2 = 0
    xy3 = 0
    x3y = 0
    N = 0

    for i in vids:
        iX = vertices[i][0]
        iY = vertices[i][1]
        iZ = vertices[i][2]

        x += iX
        y += iY
        z += iZ
        xy += iX * iY
        xz += iX * iZ
        yz += iY * iZ
        xyz += iX * iY * iZ
        x2 += iX * iX
        y2 += iY * iY
        x2y += iX * iX * iY
        x2z += iX * iX * iZ
        y2z += iY * iY * iZ
        xy2 += iX * iY * iY
        x3 += iX * iX * iX
        y3 += iY * iY * iY
        x4 += iX * iX * iX * iX
        y4 += iY * iY * iY * iY
        x2y2 += iX * iX * iY * iY
        xy3 += iX * iY * iY * iY
        x3y += iX * iX * iX * iY
        N += 1

    MatrixLeft = np.zeros((6, 6))
    MatrixRight = np.zeros((6, 1))

    MatrixLeft[0][0] = N
    MatrixLeft[0][1] = x
    MatrixLeft[0][2] = y
    MatrixLeft[0][3] = x2
    MatrixLeft[0][4] = xy
    MatrixLeft[0][5] = y2
    MatrixLeft[1][0] = x
    MatrixLeft[1][1] = x2
    MatrixLeft[1][2] = xy
    MatrixLeft[1][3] = x3
    MatrixLeft[1][4] = x2y
    MatrixLeft[1][5] = xy2
    MatrixLeft[2][0] = y
    MatrixLeft[2][1] = xy
    MatrixLeft[2][2] = y2
    MatrixLeft[2][3] = x2y
    MatrixLeft[2][4] = xy2
    MatrixLeft[2][5] = y3
    MatrixLeft[3][0] = x2
    MatrixLeft[3][1] = x3
    MatrixLeft[3][2] = x2y
    MatrixLeft[3][3] = x4
    MatrixLeft[3][4] = x3y
    MatrixLeft[3][5] = x2y2
    MatrixLeft[4][0] = xy
    MatrixLeft[4][1] = x2y
    MatrixLeft[4][2] = xy2
    MatrixLeft[4][3] = x3y
    MatrixLeft[4][4] = x2y2
    MatrixLeft[4][5] = xy3
    MatrixLeft[5][0] = y2
    MatrixLeft[5][1] = xy2
    MatrixLeft[5][2] = y3
    MatrixLeft[5][3] = x2y2
    MatrixLeft[5][4] = xy3
    MatrixLeft[5][5] = y4

    MatrixRight[0][0] = z
    MatrixRight[1][0] = xz
    MatrixRight[2][0] = yz
    MatrixRight[3][0] = x2z
    MatrixRight[4][0] = xyz
    MatrixRight[5][0] = y2z

    MatrixOut = np.dot(np.linalg.inv(MatrixLeft), MatrixRight)
    A = MatrixOut[0][0]
    B = MatrixOut[1][0]
    C = MatrixOut[2][0]
    D = MatrixOut[3][0]
    E = MatrixOut[4][0]
    F = MatrixOut[5][0]
    return A, B, C, D, E, F


def get_curvature(vertices, vert_nei_vert, nei_k=1):
    nei_sets, _ = get_nei_k(vert_nei_vert, nei_k)
    H_list = []
    K_list = []
    k1_list = []
    k2_list = []
    shape_index_list = []
    curvedness_index_list = []
    principle_direction1 = []
    principle_direction2 = []
    for i in range(vertices.shape[0]):
        A, B, C, D, E, F = fit_quad_surface(vertices, nei_sets[i])
        dx = B + 2 * D * vertices[i][0] + E * vertices[i][1]
        dy = C + E * vertices[i][0] + 2 * F * vertices[i][1]
        dxy = E
        dxx = 2 * D
        dyy = 2 * F
        # Mean
        H = (((1 + dy**2) * dxx) - (2 * dx * dy * dxy) +
             ((1 + dx**2) * dyy)) / (2 * ((1 + dx**2 + dy**2)**1.5))
        # Gaussian
        K = (dxx * dyy - dxy**2) / ((1 + dx**2 + dy**2)**2)
        k1 = H + math.sqrt(H**2 - K)
        k2 = H - math.sqrt(H**2 - K)
        shape_index = 0.5 - (1 / math.pi) * math.atan((k1 + k2) / (k1 - k2))
        curvedness_index = math.sqrt(k1**2 + k2**2) / 2

        H_list.append(H)
        K_list.append(K)
        k1_list.append(k1)
        k2_list.append(k2)
        shape_index_list.append(shape_index)
        curvedness_index_list.append(curvedness_index)

        surf_e = 1 + dx * dx
        surf_f = dx * dy
        surf_g = 1 + dy * dy
        surf_eg_f2 = surf_e + surf_g - 1
        surf_sqrt_eg_f2 = math.sqrt(surf_eg_f2)
        surf_l = dxx / surf_sqrt_eg_f2
        surf_m = dxy / surf_sqrt_eg_f2
        surf_n = dyy / surf_sqrt_eg_f2
        surf_aa = surf_n * surf_f - surf_m * surf_g
        surf_bb = surf_n * surf_e - surf_g * surf_l
        surf_cc = surf_m * surf_e - surf_f * surf_l
        surf_sqrt_b2_4ac = math.sqrt(surf_bb * surf_bb - 4 * surf_aa * surf_cc)
        surf_t1 = (-surf_bb - surf_sqrt_b2_4ac) / (2 * surf_aa)
        surf_t2 = (-surf_bb + surf_sqrt_b2_4ac) / (2 * surf_aa)
        prin_dir_x1 = 1 / math.sqrt(1 + surf_t1**2 + (dx+surf_t1*dy)**2)
        prin_dir_y1 = prin_dir_x1 * surf_t1
        prin_dir_z1 = prin_dir_x1 * dx + prin_dir_y1 * dy
        prin_dir_x2 = 1 / math.sqrt(1 + surf_t2**2 + (dx+surf_t2*dy)**2)
        prin_dir_y2 = prin_dir_x2 * surf_t2
        prin_dir_z2 = prin_dir_x2 * dx + prin_dir_y2 * dy

        principle_direction1.append([prin_dir_x1, prin_dir_y1, prin_dir_z1])
        principle_direction2.append([prin_dir_x2, prin_dir_y2, prin_dir_z2])

    return np.array(H_list), np.array(K_list), np.array(k1_list), np.array(k2_list), np.array(shape_index_list), np.array(curvedness_index_list), np.array(principle_direction1), np.array(principle_direction2)


