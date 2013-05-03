import numpy as np
import myFunc
from math import hypot, pi, cos, sin 

def non_maximal_edge_suppresion(mag, orient):
    """Non Maximal suppression of gradient magnitude and orientation."""
    # bin orientations into 4 discrete directions
    abin = ((orient + pi) * 4 / pi + 0.5).astype('int') % 4
    mask = np.zeros(mag.shape, dtype='bool')
    mask[1:-1,1:-1] = True
    edge_map = np.zeros(mag.shape, dtype='bool')
    offsets = ((1,0), (1,1), (0,1), (-1,1))
    for a, (di, dj) in zip(range(4), offsets):
        cand_idx = np.nonzero(np.logical_and(abin==a, mask))
        for i,j in zip(*cand_idx):
            if mag[i,j] > mag[i+di,j+dj] and mag[i,j] > mag[i-di,j-dj]:
                edge_map[i,j] = True
    return edge_map

def canny(im):
    #blurring the image
    # with median filter of 5*5
    temp = myFunc.medFilterSimple(im, 2)
    # High pass derivative based filters
    # with sobel operator
    Kx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    Gx = myFunc.conv2(temp, Kx)
    Gy = myFunc.conv2(temp, np.transpose(Kx))

    Grad = np.hypot(Gx, Gy)
    theta = np.arctan2(Gx, Gy)
    edge_map = non_maximal_edge_suppresion(Grad,theta)
    
    edge_map = np.logical_and(edge_map, Grad > 50)
    m, n = im.shape
    gnh = np.zeros_like(edge_map)
    gnl = np.zeros_like(edge_map)
    th, tl  = 0.2 * np.max(Grad), 0.1 * np.max(Grad)
    for x in range(1,m-1):
        for y in range(1,n-1):
            if Grad[x][y]>=th:
                gnh[x][y]=edge_map[x][y]
            if Grad[x][y]>=tl:
                gnl[x][y]=edge_map[x][y]
    gnl = gnl-gnh
    def traverse(i, j):
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if gnh[i+x[k]][j+y[k]]==0 and gnl[i+x[k]][j+y[k]]!=0:
                gnh[i+x[k]][j+y[k]]=1
                traverse(i+x[k], j+y[k])
    for i in range(1, m-1):
        for j in range(1, n-1):
            if gnh[i][j]:
                gnh[i][j]=1
                traverse(i, j)
    return np.uint8(gnh) * 255