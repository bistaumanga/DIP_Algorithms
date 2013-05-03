import cmath
import numpy as np
from math import log, ceil

def omega(p, q):
   return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def pad(lst):
   k = 0
   while 2**k < len(lst):
      k += 1
   return np.concatenate((lst, ([0] * (2**k - len(lst)))))

def fft(x):
   n = len(x)
   if n == 1:
      return x
   Feven, Fodd = fft(x[0::2]), fft(x[1::2])
   combined = [0] * n
   for m in xrange(n/2):
	   combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
	   combined[m + n/2] = Feven[m] - omega(n, -m) * Fodd[m]
   return combined

def ifft(X):
   x = fft([x.conjugate() for x in X])
   return [x.conjugate()/len(X) for x in x]

def pad2(x):
   m, n = np.shape(x)
   M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
   F = np.zeros((M,N), dtype = x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def fft2(f):
   f, m, n = pad2(f)
   return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
   f, M, N = fft2(np.conj(F))
   f = np.matrix(np.real(np.conj(f)))/(M*N)
   return f[0:m, 0:n]

def fftshift(F):
   M, N = F.shape
   R1, R2 = F[0: M/2, 0: N/2], F[M/2: M, 0: N/2]
   R3, R4 = F[0: M/2, N/2: N], F[M/2: M, N/2: N]
   sF = np.zeros(F.shape,dtype = F.dtype)
   sF[M/2: M, N/2: N], sF[0: M/2, 0: N/2] = R1, R4
   sF[M/2: M, 0: N/2], sF[0: M/2, N/2: N]= R3, R2
   return sF