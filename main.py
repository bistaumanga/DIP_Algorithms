# Implemenation of DIP algorithms
# Umanga Bista, 066BCT547, IoE Pulchowk
#
# Dependencies : 1. Python 2.7 , x86
#                2. Numpy 1.6.2 , x86
#                3. Matplotlib 1.2.0 , x86 for plotting
#                4. PIL 1.1.7 , x86 for Image I/O


import Tkinter as tk
import Image, ImageTk
import numpy as np
import tkFileDialog, tkMessageBox
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from timeit import default_timer as ticToc
import myFFT, myThresh, myHist, myCanny, myFunc
from math import hypot, pi, cos, sin 

# Everything goes inside this
class DIP(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("DIP Algorithms- Simple Photo Editor")
        self.pack(fill = tk.BOTH, expand = 1)

        menubar = tk.Menu(self.parent)
        self.parent.config(menu = menubar)

        # Initialize Labels
        self.label1 = tk.Label(self, border = 25)
        self.label2 = tk.Label(self, border = 25)
        self.label1.grid(row = 1, column = 1)
        self.label2.grid(row = 1, column = 2)

        # File Menu
        fileMenu = tk.Menu(menubar, tearoff = 0, bg = "white")
        menubar.add_cascade(label = "File", menu = fileMenu)
        # Menu Item for Open Image
        fileMenu.add_command(label = "Open", command = self.onOpen)
        # Menu Item for saving the eited image
        fileMenu.add_command(label = "Save", command = self.onSave)
        # Menu Item for Reverting back to original Image
        fileMenu.add_command(label = "Revert", command = self.setImage)
        fileMenu.add_separator()
        # Menu Item for Reverting back to original Image
        fileMenu.add_command(label = "Exit", command = self.parent.quit)

        # Basic menu
        basicMenu = tk.Menu(menubar, tearoff = 0, bg = "white")
        menubar.add_cascade(label = "Basic", menu = basicMenu)
        # Menu Item for image negative
        basicMenu.add_command(label = "Grayscale", command = self.onGryscl)
        # Menu Item for image negative
        basicMenu.add_command(label = "Negative", command = self.onNeg)
        basicMenu.add_separator()
        # Menu Item for brightness
        basicMenu.add_command(label = "Brightness", command = self.onBrghtness)
        # Menu Item for Contrast
        basicMenu.add_command(label = "Contrast", command = self.onContrast)
        # Menu item for gamma correction
        basicMenu.add_command(label = "Gamma Trans.", command = self.onGamma)
        basicMenu.add_separator()
        # Menu item for Bit Plane Extraction
        basicMenu.add_command(label = "Bit-Plane", command = self.onBitplane)

        # Histogram menu
        HistMenu = tk.Menu(menubar, tearoff = 0, bg = "white")
        menubar.add_cascade(label = "Histogram", menu = HistMenu)
        # Menu item for View Histogram
        HistMenu.add_command(label = "View Hist.", command = self.onViewHist)
        # Menu item for Histogram Equilization
        HistMenu.add_command(label = "Histogram Eq.", command = self.onHisteq)
        # Menu item for Otsu's Thresholding
        HistMenu.add_separator()
        HistMenu.add_command(label = "Otsu's Thresh", command = self.onOtsu)
        # Menu item for Global Thresholding
        HistMenu.add_command(label = "Global Thresh", command = self.onGbThresh)

        # Spatial Domain Processing menu
        spMenu = tk.Menu(menubar, tearoff = 0, bg = "white")
        menubar.add_cascade(label = "Spatial proc", menu = spMenu)
        # Menu item for averaging filter
        spMenu.add_command(label = "Averaging (LP)", command = self.onAvgLP)
        # Menu item for Weighted Average filter
        spMenu.add_command(label = "Weighted Avg (LP)", command = self.onWtAvgLP)
        # Menu item for Median filter
        spMenu.add_command(label = "Median Filter", command = self.onMed)
        spMenu.add_separator()
        # Menu item for Laplacian
        spMenu.add_command(label = "Laplacian Sharp 8", command = self.onLap8)
        # Menu item for Laplacian Sharpen
        spMenu.add_command(label = "Laplacian Sharp 4", command = self.onLap4)
        spMenu.add_separator()
        # Emboss
        spMenu.add_command(label = "Emboss", command = self.onEmboss)
        spMenu.add_command(label = "Emboss Subtle", command = self.onEmbSub)
        spMenu.add_command(label = "Emboss 2", command = self.onMot)

        # Frequency Domain Processing menu
        fqMenu = tk.Menu(menubar, tearoff = 0, bg = "white")
        menubar.add_cascade(label = "Fourier Domain", menu = fqMenu)
        # Menu item for Laplacian Sharpen
        fqMenu.add_command(label = "View Spectrum", command = self.onVwSpec)
        # Menu item for Butterworth Low Pass
        fqMenu.add_command(label = "ButterWorth LP", command = self.onBwLp)
        # Menu item for Butterworth High Pass
        fqMenu.add_command(label = "ButterWorth HP", command = self.onBwHp)
        fqMenu.add_separator()
        # Menu item for Gaussian Low Pass
        fqMenu.add_command(label = "Gaussian LP", command = self.onGaussianLp)
        # Menu item for Butterworth High Pass
        fqMenu.add_command(label = "Gaussian HP", command = self.onGaussianHp)

        # Edges menu
        edgeMenu = tk.Menu(menubar, tearoff = 0, bg = "white")
        menubar.add_cascade(label = "Edges", menu = edgeMenu)
        # Menu item for Scharr
        edgeMenu.add_command(label = "Edges Scharr", command = self.onScharr)
        # Menu item for Pewitt
        edgeMenu.add_command(label = "Edges Pewitt", command = self.onPew)
        # Menu item for Sobel
        edgeMenu.add_command(label = "Edges Sobel", command = self.onSobel)
        # Menu item for Canny
        edgeMenu.add_command(label = "Canny", command = self.onCanny)
        edgeMenu.add_command(label = "Laplacian", command = self.onLaped)
        edgeMenu.add_separator()

    def onCanny(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            self.onGryscl()
        self.I2 = self.Ilast
        temp = myCanny.canny(self.I2)
        self.onPanel2(temp)

    def onBitplane(self):
        self.I2 = self.Ilast
        temp = self.I2
        if self.Ilast.ndim == 3 :
            #tkMessageBox.showinfo("Message", "Image will be converted to Grayscale.")
            self.onGryscl()
        self.I2 = self.Ilast
        plt.clf()
        fig = plt.figure(figsize=(16,9),facecolor='w')
        fig.add_subplot(121)
        plt.subplots_adjust(left = 0, bottom=0.06, right = 1, top = 0.95, hspace = 0.1, wspace = 0)
        plt.imshow(self.I2, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Original")

        fig.add_subplot(122)
        plt.imshow(np.bitwise_and(self.I2, 128), cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Bit 8")

        axcolor = 'lightgoldenrodyellow'
        axBIT = fig.add_axes([0.2, 0.025, 0.6, 0.025], axisbg=axcolor)
        sBit = Slider(axBIT, 'Bit', 1, 8, valinit=8, valfmt='%1d')

        def update(val):
            bit = int(sBit.val)
            fig.add_subplot(122)
            plt.imshow(np.bitwise_and(self.I2, 1<<(bit - 1)), cmap = plt.cm.gray)
            plt.yticks([])
            plt.xticks([])
            msg = "bit : " + str(bit)
            plt.title(msg)

        sBit.on_changed(update)
        fig.show()
        self.onPanel2(temp)

    def onGaussianHp(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            tkMessageBox.showinfo("Message", "Image will be converted to Grayscale.")
            self.I2 = myThresh.gray(self.I2)
        #tempI, P, Q = pad2(self.I2)
        tempI = np.float64(self.I2)
        F,p,q = myFFT.fft2(tempI)
        P, Q = F.shape
        F = myFFT.fftshift(F)

        sigma = np.min([P,Q]) / 5
        H = 1.0 - myFunc.getGaussianlp(P, Q, sigma)
        temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
        temp = self.I2 + temp
        np.putmask(temp, temp > 255, 255) # check overflow
        np.putmask(temp, temp < 0, 0) # check underflow
        temp = np.uint8(np.abs(temp))
        self.onPanel2(temp)

        plt.clf()
        fig = plt.figure(figsize=(16,9),facecolor='w')
        fig.add_subplot(221)
        plt.subplots_adjust(left = 0, bottom=0.06, right = 1, top = 0.95, hspace = 0.1, wspace = 0)
        plt.imshow(self.I2, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Original Image")

        fig.add_subplot(222)
        pwrSpec = np.log10(1 + np.abs(F) ** 2)
        plt.imshow(pwrSpec, cmap = plt.cm.PRGn)
        plt.yticks([])
        plt.xticks([])
        plt.title("Power Spectrum")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Spectral Power')

        fig.add_subplot(223)
        plt.imshow(temp, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Filtered Image")

        fig.add_subplot(224)
        plt.imshow(H, cmap = plt.cm.gist_heat)
        plt.yticks([])
        plt.xticks([])
        plt.title("Frequency Response")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Amplitude')

        axcolor = 'lightgoldenrodyellow'
        axSigma = fig.add_axes([0.2, 0.025, 0.5, 0.025], axisbg=axcolor)

        sSigma = Slider(axSigma, 'Cut Off freq.( Sigma)', 5, np.min([P / 50 * 20, Q / 50 * 20]), valinit=sigma)

        def update(val):
            H = 1.0 - myFunc.getGaussianlp(P, Q, int(sSigma.val))
            temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)), p, q)
            temp = self.I2 + temp
            np.putmask(temp, temp > 255, 255) # check overflow
            np.putmask(temp, temp < 0, 0) # check underflow
            temp = np.uint8(np.abs(temp))
            self.onPanel2(temp)
            fig.add_subplot(223)

            plt.imshow(temp, cmap = plt.cm.gray)
            plt.yticks([])
            plt.xticks([])
            plt.title("Filtered Image")

            fig.add_subplot(224)
            plt.imshow(H, cmap = plt.cm.gist_heat)
            plt.yticks([])
            plt.xticks([])
            plt.title("Frequency Response")

        sSigma.on_changed(update)
        fig.show()

    def onGaussianLp(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            tkMessageBox.showinfo("Message", "Image will be converted to Grayscale.")
            self.I2 = myThresh.gray(self.I2)
        #tempI, P, Q = pad2(self.I2)
        tempI = np.float64(self.I2)
        F,p,q = myFFT.fft2(tempI)
        P, Q = F.shape
        F = myFFT.fftshift(F)

        sigma = np.min([P,Q])/10
        H = myFunc.getGaussianlp(P, Q, sigma)
        temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
        #temp = unpad2(temp, P, Q)
        temp = np.uint8(np.abs(temp))
        self.onPanel2(temp)

        plt.clf()
        fig = plt.figure(figsize=(16,9),facecolor='w')
        fig.add_subplot(221)
        plt.subplots_adjust(left = 0, bottom=0.06, right = 1, top = 0.95, hspace = 0.1, wspace = 0)
        plt.imshow(self.I2, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Original Image")

        fig.add_subplot(222)
        pwrSpec = np.log10(1 + np.abs(F) ** 2)
        plt.imshow(pwrSpec, cmap = plt.cm.PRGn)
        plt.yticks([])
        plt.xticks([])
        plt.title("Power Spectrum")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Spectral Power')

        fig.add_subplot(223)
        plt.imshow(temp, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Filtered Image")

        fig.add_subplot(224)
        plt.imshow(H, cmap = plt.cm.gist_heat)
        plt.yticks([])
        plt.xticks([])
        plt.title("Frequency Response")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Amplitude')

        axcolor = 'lightgoldenrodyellow'
        axSigma = fig.add_axes([0.2, 0.025, 0.5, 0.025], axisbg=axcolor)

        sSigma = Slider(axSigma, 'Cut Off freq.( Sigma)', 5, np.min([P / 100 * 20, Q / 100 * 20]), valinit=sigma)

        def update(val):
            H = myFunc.getGaussianlp(P, Q, int(sSigma.val))
            temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
            #temp = unpad2(temp, P, Q)
            temp = np.uint8(np.abs(temp))
            self.onPanel2(temp)
            fig.add_subplot(223)

            plt.imshow(temp, cmap = plt.cm.gray)
            plt.yticks([])
            plt.xticks([])
            plt.title("Filtered Image")

            fig.add_subplot(224)
            plt.imshow(H, cmap = plt.cm.gist_heat)
            plt.yticks([])
            plt.xticks([])
            plt.title("Frequency Response")

        sSigma.on_changed(update)
        fig.show()

    def onBwHp(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            tkMessageBox.showinfo("Message", "Image will be converted to Grayscale.")
            self.I2 = myThresh.gray(self.I2)
        #tempI, P, Q = pad2(self.I2)
        tempI = np.float64(self.I2)
        F,p,q = myFFT.fft2(tempI)
        P, Q = F.shape
        F = myFFT.fftshift(F)
        n = 10
        D0 = np.min([P,Q])/4
        H = 1.0 - myFunc.getBWlp(P, Q, n, D0)
        temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
        temp = self.I2 + temp#unpad2(temp, P, Q)
        np.putmask(temp, temp > 255, 255) # check overflow
        np.putmask(temp, temp < 0, 0) # check underflow
        temp = np.uint8(np.abs(temp))
        self.onPanel2(temp)

        plt.clf()
        fig = plt.figure(figsize=(16,9),facecolor='w')
        fig.add_subplot(221)
        plt.subplots_adjust(left = 0, bottom=0.06, right = 1, top = 0.95, hspace = 0.1, wspace = 0)
        plt.imshow(self.I2, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Original Image")

        fig.add_subplot(222)
        pwrSpec = np.log10(1 + np.abs(F) ** 2)
        plt.imshow(pwrSpec, cmap = plt.cm.PRGn)
        plt.yticks([])
        plt.xticks([])
        plt.title("Power Spectrum")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Spectral Power')

        fig.add_subplot(223)
        plt.imshow(temp, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Filtered Image")

        fig.add_subplot(224)
        plt.imshow(H, cmap = plt.cm.gist_heat)
        plt.yticks([])
        plt.xticks([])
        plt.title("Frequency Response")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Amplitude')

        axcolor = 'lightgoldenrodyellow'
        axD0 = fig.add_axes([0.1, 0.025, 0.3, 0.025], axisbg=axcolor)
        axn  = fig.add_axes([0.6, 0.025, 0.3, 0.025], axisbg=axcolor)

        sD0 = Slider(axD0, 'Cut Off freq.', 20, np.min([P / 40 * 20, Q / 40 * 20]), valinit=D0)
        sn = Slider(axn, "degree 'n'", 1, 25, valinit=n)

        def update(val):
            H = 1.0 - myFunc.getBWlp(P, Q, int(sn.val), int(sD0.val))
            temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
            temp = self.I2 + temp#unpad2(temp, P, Q)
            np.putmask(temp, temp > 255, 255) # check overflow
            np.putmask(temp, temp < 0, 0) # check underflow
            temp = np.uint8(np.abs(temp))
            self.onPanel2(temp)

            fig.add_subplot(223)
            plt.imshow(temp, cmap = plt.cm.gray)
            plt.yticks([])
            plt.xticks([])
            plt.title("Filtered Image")

            fig.add_subplot(224)
            plt.imshow(H, cmap = plt.cm.gist_heat)
            plt.yticks([])
            plt.xticks([])
            plt.title("Frequency Response")

        sD0.on_changed(update)
        sn.on_changed(update)
        fig.show()

    def onBwLp(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            tkMessageBox.showinfo("Message", "Image will be converted to Grayscale.")
            self.I2 = myThresh.gray(self.I2)
        #tempI, P, Q = pad2(self.I2)
        tempI = np.float64(self.I2)
        F,p,q = myFFT.fft2(tempI)
        P, Q = F.shape
        F = myFFT.fftshift(F)
        n = 10
        D0 = np.min([P,Q])/4
        H = myFunc.getBWlp(P, Q, n, D0)
        temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
        #temp = unpad2(temp, P, Q)
        temp = np.uint8(np.abs(temp))
        self.onPanel2(temp)

        plt.clf()
        fig = plt.figure(figsize=(16,9),facecolor='w')
        fig.add_subplot(221)
        plt.subplots_adjust(left = 0, bottom=0.06, right = 1, top = 0.95, hspace = 0.1, wspace = 0)
        plt.imshow(self.I2, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Original Image")

        fig.add_subplot(222)
        pwrSpec = np.log10(1 + np.abs(F) ** 2)
        plt.imshow(pwrSpec, cmap = plt.cm.PRGn)
        plt.yticks([])
        plt.xticks([])
        plt.title("Power Spectrum")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Spectral Power')

        fig.add_subplot(223)
        plt.imshow(temp, cmap = plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.title("Filtered Image")

        fig.add_subplot(224)
        plt.imshow(H, cmap = plt.cm.gist_heat)
        plt.yticks([])
        plt.xticks([])
        plt.title("Frequency Response")
        cbar = plt.colorbar(ticks = [])
        cbar.set_label(r'Amplitude')

        axcolor = 'lightgoldenrodyellow'
        axD0 = fig.add_axes([0.1, 0.025, 0.3, 0.025], axisbg=axcolor)
        axn  = fig.add_axes([0.6, 0.025, 0.3, 0.025], axisbg=axcolor)

        sD0 = Slider(axD0, 'Cut Off freq.', 20, np.min([P / 40 * 20, Q / 40 * 20]), valinit=D0)
        sn = Slider(axn, "degree 'n'", 1, 25, valinit=n)

        def update(val):
            H = myFunc.getBWlp(P, Q, int(sn.val), int(sD0.val))
            temp = myFFT.ifft2(myFFT.fftshift(np.multiply(F , H)),p,q)
            #temp = unpad2(temp, P, Q)
            temp = np.uint8(np.abs(temp))
            self.onPanel2(temp)

            fig.add_subplot(223)
            plt.imshow(temp, cmap = plt.cm.gray)
            plt.yticks([])
            plt.xticks([])
            plt.title("Filtered Image")

            fig.add_subplot(224)
            plt.imshow(H, cmap = plt.cm.gist_heat)
            plt.yticks([])
            plt.xticks([])
            plt.title("Frequency Response")

        sD0.on_changed(update)
        sn.on_changed(update)
        fig.show()

    def onVwSpec(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            tkMessageBox.showinfo("Message", "Image will be converted to Grayscale.")
            self.I2 = myThresh.gray(self.I2)
        # padding before taking Fourier Transoffrm
        #temp, _, _ = pad2(self.I2)
        #temp = self.I2
        temp = np.float64(self.I2)
        F,_,_ = myFFT.fft2(temp)
        F = myFFT.fftshift(F)
        pwrSpec = np.log10(1 + np.abs(F) ** 2)
        plt.imshow(pwrSpec, cmap = plt.cm.PRGn)
        plt.yticks([])
        plt.xticks([])
        plt.title("Power Spectrum")
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('Frequency Amplitudes')
        plt.show()

    def onLaped(self):
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            self.onGryscl()
        self.spKer(kernel)

    def onPew(self):
        Kx = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
        self.spEdges(Kx)

    def onSobel(self):
        Kx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        self.spEdges(Kx)

    def onScharr(self):
        Kx = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
        self.spEdges(Kx)

    def spEdges(self, Kx) :
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            self.onGryscl()
        self.I2 = self.Ilast
        tempX = myFunc.conv2(self.I2, Kx)
        tempY = myFunc.conv2(self.I2, np.transpose(Kx))
        temp = np.abs(tempX) + np.abs(tempY) # calculate the new image
        self.onPanel2(temp)

    def onMot(self):
        kernel = [[2, 0, 0], [0,-1,0], [0,0,-1]]
        self.spKer(kernel, 127)

    def onEmboss(self):
        kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
        self.spKer(kernel, 127)

    def onEmbSub(self):
        kernel = [[1,1,-1], [1, 3, -1], [1,-1,-1]]
        self.spKer(kernel, 127)

    def onLap8(self):
        kernel = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
        self.spKer(kernel)

    def onLap4(self):
        kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        self.spKer(kernel)

    def spKer(self, kernel,offset = 0):
        self.I2 = self.Ilast
        kernel = np.float32(kernel)
        self.I2= np.float32(self.I2)
        if self.Ilast.ndim == 3 :
            temp = np.zeros(self.I2.shape, dtype = self.I2.dtype)
            temp[:,:,0] = myFunc.conv2(self.I2[:,:,0], kernel)
            temp[:,:,1] = myFunc.conv2(self.I2[:,:,1], kernel)
            temp[:,:,2] = myFunc.conv2(self.I2[:,:,2], kernel)
        else:
            temp = myFunc.conv2(self.I2, kernel)
        np.putmask(temp, temp > 255.0, 255.0) # overfloe check
        np.putmask(temp, temp < 0.0, 0.0) # check underflow
        self.onPanel2(np.uint8(temp))

    # Median Filter
    def onMed(self):
        self.I2 = self.Ilast
        if self.Ilast.ndim == 3 :
            tkMessageBox.showinfo("Message", "This operation is slow ! \n Image will be converted to Grayscale.")
            self.onGryscl()
        self.I2 = self.Ilast
        self.onMedKsize(3)
#        tempTk = tk.Tk()
#        tempTk.geometry("320x120")
#        tempTk.title("Block Size")
#        tmpLbl = tk.Label(tempTk, text = """Please Select the Block Size via slider.
#                    Default Block Size is '5 * 5'\n\n""", font=('Arial',12))
#        tmpLbl.pack()
#        tempSc = tk.Scale( tempTk, from_ = 1, to = 5, orient = tk.HORIZONTAL, showvalue = 0,
#                          command = self.onMedKsize, length = 200 ,width = 10, sliderlength = 15)
#        tempSc.set(2)
#        tempSc.pack(anchor = tk.CENTER)

    def onMedKsize(self, new_value):
        tic = ticToc()
        s = int(new_value)
        temp = myFunc.medFilterSimple(self.I2, s/2)
        self.onPanel2(temp)
        toc = ticToc()
        print toc - tic

    # Weighted Averaging Filter
    def onWtAvgLP(self):
        self.I2 = self.Ilast
        kernel = 1.0 / 16 * np.float32([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        self.spKer(kernel)

    # for Averaging Filter
    def onAvgLP(self):
        self.I2 = self.Ilast
        kernel = 1.0 / 9 * np.ones((3, 3))
        self.spKer(kernel)

    def onGbThresh(self):
        # Convert to grayscale
        self.I2 = self.Ilast
        if self.I2.ndim == 3 :
            self.I2 = myThresh.gray(self.I2)
        #calculate Histogram
        T, h = myThresh.gbThresh(self.I2)
        temp = self.I2 >= T
        temp = 255 * temp
        # configure the 2nd label
        self.onPanel2(temp)
        plt.plot(np.arange(0, 256), h, color = 'black', linewidth = 2, label = "Histogram")
        plt.axvline(x = T, color = 'green', linewidth = 2, label = "Threshold Value")
        plt.text(T, np.max(h) / 1.2, 'T : ' + str(T), color = 'red')
        plt.legend(loc = 'upper right')
        plt.title("Global Thresholding")
        plt.xlabel("Grayscale Intensity (rk)")
        plt.ylabel("Normalized Histogram, p(rk)")
        plt.show()

    def onOtsu(self):
        # Convert to grayscale
        self.I2 = self.Ilast
        #calculate Histogram
        if self.I2.ndim == 3 :
            self.I2 = myThresh.gray(self.I2)
        thresh, h = myThresh.otsu(self.I2)
        temp = self.I2 >= thresh
        temp = 255 * temp
        # configure the 2nd label
        self.onPanel2(temp)
        #thresh = np.float(thresh)
        plt.plot(np.arange(0, 256), h, color = 'black', linewidth = 2, label = "Histogram")
        plt.axvline(x = thresh, color = 'green', linewidth = 2, label = "Threshold Value")
        plt.text(thresh, np.max(h) / 1.2, 'T : ' + str(thresh), color = 'red')
        plt.legend(loc = 'upper right')
        plt.title("Demostration of Histogram partioning in Otsu's Thresholding")
        plt.xlabel("Grayscale Intensity (rk)")
        plt.ylabel("Normalized Histogram, p(rk)")
        plt.show()

    def onHisteq(self):
        self.I2 = self.Ilast
        if self.I2.ndim == 2 :
            
            temp, h, h2, sk = myHist.histeq(self.I2)
            # configuring new image in 2nd label
            self.onPanel2(temp)

            # plot histogram of original image
            plt.subplot(221)
            markerline, stemlines, baseline = plt.stem(np.arange(0, 256), h, '-')
            plt.setp(markerline, 'marker', '.', 'markerfacecolor', 'none')
            plt.setp(stemlines, 'color', 'b')
            plt.setp(baseline, 'color', 'black', 'linewidth', 3)
            plt.plot(np.arange(0, 256), h, color = 'black', linewidth = 2)
            plt.title("Histogram of Original Image")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")

            # plot histogram of Equalized Image
            plt.subplot(223)
            markerline, stemlines, baseline = plt.stem(np.arange(0, 256), h2, '-')
            plt.setp(markerline, 'marker', '.', 'markerfacecolor', 'none')
            plt.setp(stemlines, 'color', 'b')
            plt.setp(baseline, 'color','black', 'linewidth', 3)
            plt.title("Equalized Histogram")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")

            # plot to show transfer function
            plt.subplot(222)
            plt.plot(np.arange(0, 256), sk, color = 'green', linewidth = 3, label = 'Red Intensities')
            plt.title(' Transfer Function')
            plt.xlabel("rk")
            plt.ylabel("sk = f(rk)")

            plt.show()

        else :
            r, g, b = self.I2[:,:,0], self.I2[:,:,1], self.I2[:,:,2]
            R, hR, hR2, skR = myHist.histeq(r)
            G, hG, hG2, skG = myHist.histeq(g)
            B, hB, hB2, skB = myHist.histeq(b)
            temp = np.zeros_like(self.I2)
            temp[:,:,0], temp[:,:,1], temp[:,:,2] = R, G, B
            # configuring new image in 2nd label
            self.onPanel2(temp)

            # plot histogram of original image
            plt.subplot(221)

            plt.subplot(221)
            # for first histogram
            plt.plot(np.arange(0, 256), hR, color = 'red', linewidth = 2, label = 'Red Intensities')
            plt.plot(np.arange(0, 256), hG, color = 'green', linewidth = 2, label = 'Blue Intensities')
            plt.plot(np.arange(0, 256), hB, color = 'blue', linewidth = 2, label = 'Green Intensities')
            #plt.legend(loc='upper left')
            plt.title("Histogram of Original Image")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")

            # for second histogram
            plt.subplot(223)
            plt.plot(np.arange(0, 256), hR2, color = 'red', linewidth = 2, label = 'Red Intensities')
            plt.plot(np.arange(0, 256), hG2, color = 'green', linewidth = 2, label = 'Blue Intensities')
            plt.plot(np.arange(0, 256), hB2, color = 'blue', linewidth = 2, label = 'Green Intensities')
            #plt.legend(loc='upper left')
            plt.title("Histogram of Equalized Image")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")

            #for transfer function
            plt.subplot(222)
            plt.plot(np.arange(0, 256), skR, color = 'red', linewidth = 2, label = 'Red Intensities')
            plt.plot(np.arange(0, 256), skG, color = 'green', linewidth = 2, label = 'Blue Intensities')
            plt.plot(np.arange(0, 256), skB, color = 'blue', linewidth = 2, label = 'Green Intensities')
            #plt.legend(loc='upper left')
            plt.title("Transfer Function")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")
            plt.show()

    def onViewHist(self):
        self.I2 = self.Ilast
        if self.I2.ndim==2 :
            # calculate the histogram
            h = myHist.imhist(self.I2)
            # and show the histogram
            markerline, stemlines, baseline = plt.stem(np.arange(0, 256), h, '-')
            plt.setp(markerline, 'marker', '.', 'markerfacecolor', 'none')
            plt.setp(stemlines, 'color', 'b')
            plt.setp(baseline, 'color','black', 'linewidth', 3)
            plt.plot(np.arange(0, 256), h, color = 'black', linewidth = 2)
            plt.title("Histogram")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")
            plt.show()
        else :
            # calculate the histogram for each R, G and B components
            hR = myHist.imhist( self.I2[:, :, 0])
            hG = myHist.imhist( self.I2[:, :, 1])
            hB = myHist.imhist( self.I2[:, :, 2])
            # and show the output
            plt.plot(np.arange(0, 256), hR, color = 'red', linewidth = 2, label = 'Red Intensities')
            plt.plot(np.arange(0, 256), hG, color = 'green', linewidth = 2, label = 'Blue Intensities')
            plt.plot(np.arange(0, 256), hB, color = 'blue', linewidth = 2, label = 'Green Intensities')
            plt.legend(loc='upper left')
            plt.title("Histogram")
            plt.xlabel("Intensity Values (8 bit) rk")
            plt.ylabel("p(rk)")
            plt.show()

    def onGamma(self):
        # gamma Transformations
        tempTk = tk.Tk()
        self.I2 = self.Ilast
        tempSc = tk.Scale( tempTk, from_ = 0.2, to = 5, resolution = 0.1, orient = tk.HORIZONTAL,
                            command = self.adjCntrst, length = 200 ,width = 10, sliderlength = 15)
        tempSc.set(1.0) #set initial slider value to 1.0
        tempSc.pack(anchor = tk.CENTER)

    def adjGamma(self, new_value):
        new_value = int(new_value)
        temp = np.uint16(self.I2)
        # apply the transfer function
        temp = temp ** new_value
        # check overflow and underflow
        np.putmask(temp, temp > 255, 255)
        np.putmask(temp, temp < 0, 0)
        temp = np.uint8(temp)
        # configure the 2nd label
        self.onPanel2(temp)

    def onGryscl(self):
        # Convert to grayscale
        self.I2 = self.Ilast
        if self.Ilast.ndim == 2 :
            tkMessageBox.showinfo("Message", "Image is already Grayscale")
        else :
            s = np.shape(self.Ilast)
            temp = np.zeros((s[0],s[1]),dtype=np.uint16)
            # calculate grayvalue by average of R, G and B
            self.Ilast = np.float64(self.Ilast)
            temp = myThresh.gray(self.Ilast)
            temp = np.uint8(temp)
            # configure the 2nd label
            self.onPanel2(temp)

    def onBrghtness(self):
        #Image Brightness Adjustment Menu callback
        tempTk = tk.Tk()
        self.I2 = self.Ilast
        tempSc = tk.Scale( tempTk, from_ = -100, to = 100, orient = tk.HORIZONTAL,
                          command = self.adjBright, length = 200 ,width = 10, sliderlength = 15)
        tempSc.pack(anchor = tk.CENTER)

    def adjBright(self, new_value):
        new_value = int(new_value)
        temp = np.uint16(self.I2)
        temp = temp + new_value # add/subtract the value
        np.putmask(temp, temp > 255, 255) # check overflow
        np.putmask(temp, temp < 0, 0) # check underflow
        temp = np.uint8(temp)
        # configure the 2nd label
        self.onPanel2(temp)

    def onContrast(self):
        # Image Contrast Adjustment Menu callback
        tempTk = tk.Tk()
        self.I2 = self.Ilast
        tempSc = tk.Scale( tempTk, from_ = 0.5, to = 2, resolution = 0.1, orient = tk.HORIZONTAL,
                            command = self.adjCntrst, length = 200 ,width = 10, sliderlength = 15)
        tempSc.set(1.0) # set initial factor to 1.0
        tempSc.pack(anchor = tk.CENTER)

    def adjCntrst(self, new_value):
        new_value = float(new_value)
        temp = np.float16(self.I2)
        temp = new_value * temp # aply contrast by multiplication
        np.putmask(temp, temp > 255, 255) # overfloe check
        np.putmask(temp, temp < 0, 0) # check underflow
        temp = np.uint8(temp)
        # configure the 2nd label
        self.onPanel2(temp)

    def onNeg(self):
        # Image Negative Menu callback
        self.I2 = self.Ilast
        temp = 255-self.I2; # subtract from maximum value
        temp = np.uint8(temp)
        # configure the 2nd label
        self.onPanel2(temp)

    def setImage(self):
        try:
            self.img = Image.open(self.fn)
            self.I = np.asarray(self.img)
            l, h = self.img.size
            if np.max([l,h])> 512 :
                self.img.thumbnail((512,512), Image.ANTIALIAS)
                self.I = np.asarray(self.img)
                l, h = self.img.size

            text = str(2*l+100)+"x"+str(h+50)+"+0+0"
            self.parent.geometry(text)
            photo = ImageTk.PhotoImage(self.img)
            self.label1.configure(image = photo)
            self.label1.image = photo # keep a reference!
            #for 2nd label
            self.I2 = self.I
            self.Ilast = self.I
            self.im = Image.fromarray(np.uint8(self.I2))
            photo2 = ImageTk.PhotoImage(self.im)
            self.label2.configure(image = photo2)
            self.label2.image = photo2 # keep a reference!
        except IOError as e:
            print e

    def onOpen(self):
        #Open Callback
        ftypes = [('Image Files', '*.tif *.jpg *.png ')]
        dlg = tkFileDialog.Open(self, filetypes = ftypes)
        filename = dlg.show()
        self.fn = filename
        self.setImage()

    def onSave(self):
        file_opt = options = {}
        options['filetypes'] = [('Image Files', '*.tif *.jpg *.png')]
        options['initialfile'] = 'myImage.jpg'
        options['parent'] = self.parent
        fname = tkFileDialog.asksaveasfilename(**file_opt)
        Image.fromarray(np.uint8(self.Ilast)).save(fname)

    def onPanel2(self, temp):
        self.im = Image.fromarray(temp)
        photo2 = ImageTk.PhotoImage(self.im)
        self.label2.configure(image = photo2)
        self.label2.image = photo2 # keep a reference!
        self.Ilast = temp

def main():
    root = tk.Tk()
    DIP(root)
    root.geometry("640x480")
    root.mainloop()

if __name__ == '__main__':
    main()