import myHist
import numpy as np

def gray(im):
	return np.uint8(0.2126* im[:,:,0]) + np.uint8(0.7152 * im[:,:,1]) + np.uint8(0.0722 * im[:,:,2])

def gbThresh(im):
	if im.ndim == 3 :
		im = gray(im)
	h = myHist.imhist(im)
	i = np.arange(0, 256) # all intensity values
	globalMean = np.uint8(np.sum(h* i)) # global mean
	T = globalMean #initialize threshold as globalMean
	while True:
		M1 = np.uint16(np.sum(h[0 : T] * np.arange(0, T)) / np.sum(h[0 : T]))
		M2 = np.uint16(np.sum(h[T : 255] * np.arange(T, 255)) / np.sum(h[T : 255]))
		T2 = (M1 + M2) / 2
		if abs(T2-T) <= 1:
			break
		T = T2
	return T, h

def otsu(im):
	if im.ndim == 3 :
		im = gray(im)
	h = myHist.imhist(im)

	cdf = np.array(myHist.cumsum(h)) #cumulative distribution function
	i = np.arange(0, 256) # all intensity values
	cumMean = np.array(myHist.cumsum(h * i))  #cumulative mean for all intnsities
	globalMean = np.sum(h* i) # global mean
	# calculate between class variance
	varB = np.floor(np.power(globalMean * cdf  - cumMean, 2) / (cdf * (1 - cdf)))
	# threshold k* is the value of k in varB(k) for which varB(k) is maximum
	thresh = np.where(varB == np.nanmax(varB))
	thresh = np.mean(thresh)
	return thresh, h
