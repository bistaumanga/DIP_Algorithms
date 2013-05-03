from myFFT import *
import numpy as np
x = np.matrix([[1,2,1],[2,1,2],[0,1,1]])
#y,_,_ = pad2(x)
Y, m, n = fft2(x)
print Y
print ifft2(Y, m, n)