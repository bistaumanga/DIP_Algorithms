import numpy as np
from timeit import default_timer as ticToc

# Median Filter , Simple approach
def medFilterSimple(X, r):
    y, x = np.shape(X)
    Y = np.copy(X)
    mid = (2 * r + 1) ** 2 / 2
    for i in range(r, y - r ) :
        for j in range(r, x - r):
            roi = X[i - r : i + r + 1, j - r : j + r + 1]
            lst = list(roi.flat)
            lst.sort()
            Y[i, j] = lst[mid]
    return Y

def pad2(mat) :
    # padding with zeros
    M, N = np.shape(mat)
    P, Q = 2 * M, 2 * N
    padMat = np.zeros((P,Q), dtype = mat.dtype)
    padMat[M / 2 : - M / 2, N / 2 : - N / 2] = mat
    return padMat, P, Q

def unpad2(padMat, P, Q) :
    # unpadding zeros
    M, N = P / 2, Q / 2
    return padMat[M / 2 : - M / 2, N / 2 : - N / 2]

def getBWlp(P, Q, n, D0):
    u = np.arange(-P / 2, P / 2 )
    v = np.arange(-Q / 2, Q / 2 )
    u, v = np.meshgrid(v, u)
    Duv = np.sqrt(u ** 2 + v ** 2)
    H = 1.0 / (1 + (Duv / D0) ** 2 * n)
    return H

def getGaussianlp(P, Q, sigma):
    u = np.arange(-P / 2, P / 2 )
    v = np.arange(-Q / 2, Q / 2 )
    u, v = np.meshgrid(v, u)
    Duv2 = np.sqrt(u ** 2 + v ** 2)
    H = np.exp( - Duv2 ** 2 / (2 * sigma ** 2))
    return H

# function that convolvs image with kernel
def conv2(im, kern2d):
    tic = ticToc()
    kern2d = np.float32(np.matrix(kern2d))
    kx, ky = kern2d.shape
    assert kx==ky and kx%2==1
    r = (kx-1)//2
    nl, nc = im.shape
    im = np.float32(im)
    im2 = np.zeros((nl-2*r, nc-2*r), np.float32)
    for k in xrange(-r, r+1):
        for l in xrange(-r, r+1):
            im2 += im[r+k:nl-r+k, r+l:nc-r+l] * kern2d[r+k, r+l]
    print ticToc() - tic
    im[r : nl - r, r : nc - r] = im2
    return im

def get_normalized_values(a):
    """I = Imin + (Imax-Imin)*(D-Dmin)/(Dmax-Dmin)"""
    imin = 0.0
    imax = 255.0
    dmin = np.min(a)
    dmax = np.max(a)
    normalized = imin + (imax - imin)*(a - dmin)/(dmax - dmin)
    return np.uint8(normalized)

