from copy import deepcopy
from math import exp, floor, log10, ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np

# the use of class and multiprecessing in Keras 
# will cause multiple cpus read/write self.img at the same time, which may lead to a bug

# Adaptive histogram equalization
def clahe(img, limit=10, tileGridSize=(10, 10)):
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=tileGridSize)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = clahe.apply(img[i])
    return temp

# circular mask
def circularmask(img):
    center = [int(img.shape[2]/2), int(img.shape[1]/2)]
    radius = min(center[0], center[1], img.shape[2]-center[0], img.shape[1]-center[1])
    Y, X = np.ogrid[:img.shape[1], :img.shape[2]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask_array = dist_from_center <= radius
    temp = img
    temp[:,~mask_array] = 0
    return temp

# square mask
def squaremask(img):
    '''
    get the bigest square inside image with circular mask
    will change the input size
    '''
    n = img.shape[1]
    start = ceil(n*0.5*(1.-0.5**0.5))
    end = floor(n-n*0.5*(1.-0.5**0.5))
    return img[:,start-1:end,start-1:end]

def poisson_noise(img, c=1.):
    '''
    produce poisson noise on given images
    Smaller c brings higher noise level
    '''
    temp = np.zeros_like(img)

    for i in range(img.shape[0]):
        vals = len(np.unique(img[i]))
        vals = 2 ** np.ceil(np.log2(vals))
        temp[i] = np.random.poisson(img[i] * c * vals) / float(vals) / c
    
    return temp

def bac(img, a=1, b=0):
    temp = np.clip(a*img+b, 0., 255.)
    temp = temp.astype(np.uint8)
    return temp

def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = cv2.LUT(img[i], gamma_table)
    temp = temp.astype(np.uint8)
    return temp


class imgprocess(object):
    def __init__(self, recipe):
        '''
        image dtype: ndarray, uint8
        '''
        self.shape = (0, 0)
        self.recipe = recipe

    def load_img(self, obj):
        self.img = obj
        self.shape = self.img.shape  # n*row*column

    def return_img(self):
        return self.img
    
    def batch_process(self,obj):
        self.load_img(obj)
        for i in self.recipe:
           exec('self.'+i)
        return self.img

    # def save(self, path):
    #     try:
    #         cv2.imwrite(path, self.img_p)
    #     except Exception as e:
    #         print("saving failed: %s" % path)

    def show(self):
        # htitch= np.hstack((self.img, self.img_p))
        for i in range(self.shape[0]):
            cv2.imshow('image', self.img[i])
            cv2.waitKey(200000)
            cv2.destroyAllWindows()

    def showhist(self):
        plt.figure('fig')
        plt.subplot(1,2,1)
        plt.imshow(self.img[0], 'gray')
        # plt.subplot(2,2,2)
        # plt.imshow(self.img_p, 'gray')
        plt.subplot(1,2,2)
        plt.hist(self.img[0].ravel(),256,[0,256])
        # plt.subplot(1,2,2)
        # plt.hist(self.img_p.ravel(),256,[0,256])
        plt.ion()
        plt.pause(50)
        plt.close()

    # img processing functions
    # binning
    def binning(self, bsize=(2, 2)):
        temp = np.zeros((self.shape[0], int(
            round(self.shape[1]/bsize[0])), int(round(self.shape[2]/bsize[1]))))
        for i in range(self.shape[0]):
            temp[i] = cv2.resize(self.img[i], (int(round(self.shape[2]/bsize[1])), int(
                round(self.shape[1]/bsize[0]))), interpolation=cv2.INTER_LINEAR)
        self.img = temp.astype(np.uint8)
        self.shape = temp.shape

    # brightness and contrast

    def bac(self, a=1, b=0):
        self.img = np.clip(a*self.img+b, 0., 255.)
        self.img = self.img.astype(np.uint8)

    def autobac(self):
        alpha = 255.0 / (np.amax(self.img)-np.amin(self.img))
        beta = -alpha*np.amin(self.img)
        self.bac(alpha, beta)

    # gamma
        # if gamma < 1:
        # The whole figure is brighter, contrast of dark part is increased
        # vise versa
    def gamma_trans(self, gamma):
        gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        temp = np.zeros_like(self.img)
        for i in range(self.shape[0]):
            temp[i] = cv2.LUT(self.img[i], gamma_table)
        self.img = temp

    def autogamma(self):
        meanGrayVal = np.sum(self.img) / (self.shape[0]*self.shape[1]*self.shape[2])
        gamma = log10(1/2.2)/log10(meanGrayVal/255.0)
        self.gamma_trans(gamma)

    # highpass filter through dct
    # def highpass(self, sd):
    #     row, col = self.img_p.shape[:2]

    #     # shift the zero-frequency component to the center of the spectrum
    #     fshift = np.fft.fftshift(cv2.dct(self.img_p.astype(np.float)))

    #     transfor_matrix = np.zeros(self.img_p.shape, dtype=np.float)
    #     for i in range(row):
    #         for j in range(col):
    #             transfor_matrix[i,j] = (i-row/2)**2+(j-col/2)**2
    #     transfor_matrix = 1 - np.exp(-transfor_matrix/(2*sd**2))
    #     # keep the average brightness and contrast
    #     transfor_matrix[int(floor(row/2)), int(floor(col/2))] = 1.0

    #     fback = np.round(cv2.idct(np.fft.ifftshift(fshift*transfor_matrix)))

    #     self.img_p = fback.astype(np.uint8)
    #     self.img_p = np.clip(self.img_p, 0, 255)

    # histogram equalization
    def equalization(self):
        temp = np.zeros_like(self.img)
        for i in range(self.shape[0]):
            temp[i] = cv2.equalizeHist(self.img[i])
        self.img = temp

    # Adaptive histogram equalization
    def clahe(self, limit=8.0, tileGridSize=(10, 10)):
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=tileGridSize)
        temp = np.zeros_like(self.img)
        for i in range(self.shape[0]):
            temp[i] = clahe.apply(self.img[i])
        self.img = temp

    # circular mask
    def circularmask(self):
        center = [int(self.shape[2]/2), int(self.shape[1]/2)]
        radius = min(center[0], center[1], self.shape[2]-center[0], self.shape[1]-center[1])
        Y, X = np.ogrid[:self.shape[1], :self.shape[2]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask_array = dist_from_center <= radius
        self.img[:,~mask_array] = 0
    
    # square mask
    def squaremask(self):
        '''
        get the bigest square inside image with circular mask
        will change the input size
        '''
        n = self.shape[1]
        start = ceil(n*0.5*(1.-0.5**0.5))
        end = floor(n-n*0.5*(1.-0.5**0.5))
        temp = self.img[:,start-1:end,start-1:end]
        self.img = temp
        self.shape = self.img.shape

    def noisy(self,noise_typ,image):
        '''
        contains common noise types
        return: image with noise, with same shape of input
        '''
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            #var = 0.1
            #sigma = var**0.5
            gauss = np.random.normal(mean,1,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = image
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy

    def poisson_noise(self,c=1):
        '''
        produce poisson noise on given images
        Smaller c brings higher noise level
        '''
        for i in range(self.shape[0]):
            self.img[i] = self.noisy("poisson",self.img[i]*c) / c


    # TODO: noise cancelling after equalization
    # TODO: different window functions


if __name__ == '__main__':
    pass
