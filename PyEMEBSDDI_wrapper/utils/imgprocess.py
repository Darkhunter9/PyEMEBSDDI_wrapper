"""
BSD 3-Clause License

Copyright (c) 2020, Zihao Ding, Marc De Graef Research Group/Carnegie Mellon University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from copy import deepcopy
from math import exp, floor, log10, ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def autobac(img):
    alpha = 255.0 / (np.amax(img)-np.amin(img))
    beta = -alpha*np.amin(img)
    return bac(img, alpha, beta)

def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = cv2.LUT(img[i], gamma_table)
    temp = temp.astype(np.uint8)
    return temp

def autogamma(img):
    meanGrayVal = np.sum(img) / (img.shape[0]*img.shape[1]*img.shape[2])
    gamma = log10(1/2.2)/log10(meanGrayVal/255.0)
    return gamma_trans(img, gamma)

def binning(img, bsize=(2, 2)):
    temp = np.zeros((img.shape[0], floor(img.shape[1]/bsize[0]), floor(img.shape[2]/bsize[1])))
    for i in range(img.shape[0]):
        temp[i] = cv2.resize(img[i], (floor(img.shape[2]/bsize[1]), floor(img.shape[1]/bsize[0])), interpolation=cv2.INTER_LINEAR)
    temp = temp.astype(np.uint8)
    return temp


# TODO: different window functions


if __name__ == '__main__':
    pass
