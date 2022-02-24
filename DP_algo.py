#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:26:39 2022

@author: lduguet
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from time import time
import utils as tls

def DP_shifts_on_the_grid(U,V,R,X,gamma,show=False):
    """
    solves the DP problem using a DP algorithm.

    Parameters
    ----------
    U : np.array
        Even lines serving as reference for finding the shifts.
    V : np.array
        Odd lines that must be shifted.
    R : integer
        search pattern length.
    X : integer
        grid refinement term: there is X subpixels in 1 pixel.
    gamma : float
        regularization parameter.
    show : boolean, optional
        whether or not to display solving informations (recommended only for big images). The default is False.

    Returns
    -------
    shift : np.array
        shift solving the DP problem that must be applied to dejitter the image.

    """
    n = U.shape[1]-2*R
    x = np.zeros((1,1,X))
    x[0,0,:] = np.linspace(0,1,X,endpoint=False)
    v_base = gamma*np.linspace(-R,R,2*R*X,endpoint=False)**2
    du = np.expand_dims(np.roll(U,-1,axis = 1)[1:,:-1]-U[1:,:-1],axis=2)
    t=time()
    B = [np.linalg.norm(np.reshape(np.multiply(du[:,R:2*R],x),(-1,R*X))+np.repeat(U[1:,R:2*R],X,axis = 1)-np.expand_dims((V[:-1,R]+V[1:,R])/2,axis=1),ord = 1,axis=0)]
    for i in range(1,n):
        if show and i%1000 == 200:
            print("    Constructing messages at column : ",i," - time : ",time()-t," - ",int(i/n*100),"% of processed columns")
        k = np.argmin(B[-1])
        B.append(np.linalg.norm(np.reshape(np.multiply(du[:,R+i:2*R+i],x),(-1,R*X))+np.repeat(U[1:,R+i:2*R+i],X,axis = 1)-np.expand_dims((V[:-1,i+R]+V[1:,i+R])/2,axis=1),ord = 1,axis=0)+B[-1][k]+v_base[R*X-k:2*R*X-k])
    d = [np.argmin(B[-1])]
    for i in np.arange(n-1,-1,-1):
        d.append(np.argmin(B[i]+v_base[R*X-d[-1]:2*R*X-d[-1]]))
    shift = np.zeros((U.shape[1]))
    shift[R-1:-R] = np.array(d[::-1])/X
    return shift


def dejitt(Odd,d):
    """
    Shifts the given lines according to the given shifts

    Parameters
    ----------
    Odd : np.array
        odd lines to be shifted.
    d : np.array
        shift to apply to the odd lines.

    Returns
    -------
    recal_Odd : np.array
        resulting shifted lines.

    """
    kid = np.clip(d+np.arange(len(d)),0,Odd.shape[1])[:-1]
    recal_Odd = np.zeros(Odd.shape)[:,:-1]
    for j in np.arange(Odd.shape[1]-1):
        k2 = np.argmax(kid>j)
        k1 = k2-1
        dx = kid[k2]-kid[k1]
        recal_Odd[:,j] = Odd[:,k1]/dx*(kid[k2]-j)+Odd[:,k2]/dx*(j-kid[k1])
    return recal_Odd


def main_DP_on_the_grid_L1_regul2(im,R,X,gamma,name="",save=False,memory_range=16,show=False):
    """
    Solves the DP problem with a L1 data proximity term and a L2 shift regularization term

    Parameters
    ----------
    im : np.array
        jittered image.
    R : integer
        search pattern length.
    X : integer
        grid refinement term: there is X subpixels in 1 pixel.
    gamma : float
        regularization parameter.

    Returns
    -------
    np.array
        dejittered image.
    np.array
        solution shifts.
    float
        time taken to solve the problem and reconstruct the image.

    """
    if show:
        print("Starting the DP algorithm")
    t0 = time()
    Even = im[::2,:].astype(np.float64)
    Odd = im[1::2,:].astype(np.float64)
    d = DP_shifts_on_the_grid(Even,Odd,R,X,gamma, show)
    t2 = time()
    if show:
        print("Applying the shift on the jittered image")
    new_Odd = dejitt(Odd,d)
    t1 = time()
    if show:
        print("Image reconstructed in ",t1-t2)
    dej_im = np.copy(im).astype(np.float64)
    dej_im[1::2,:-1] = new_Odd
    if save:
        if memory_range<=8:
            io.imsave("../images/dejitt_DP_"+name+".png",np.clip(dej_im[:,R:-R],0,2**memory_range-1).astype(np.uint8))
        else:
            io.imsave("../images/dejitt_DP_"+name+".png",np.clip(dej_im[:,R:-R],0,2**memory_range-1).astype(np.uint16))
        if show:
            t2 = time()
            print("Reconstructed image saved in ",t2-t1)
    t1 = time()
    if show:
        print("Total time === ",t1-t0)
    return dej_im[:,R:-R],d[R:-R],t1-t0

def demo_DP_on_the_grid(opt = 0):
    """
    Demo of main_DP_on_the_grid_L1_regul2 use for 2 possible cases (Chelsea and V_manu)
    Hand-made jitter follows lens speed type function.

    Parameters
    ----------
    opt : int, optional
        demo solve if 0 for Chelsea image, else for V_manu image . The default is 0 (Chelsea).

    Returns
    -------
    None.

    """
    if opt == 0:
        im = np.array(io.imread("../images/Chelsea.png")).astype(np.float64)
        name = "Chelsea"
        gamma = 300
    else:
        im = np.array(io.imread("../images/V_manu.png")).astype(np.float64)
        name = "V_manu"
        gamma = 10
    memory_range = 8
    true_I = 10.4
    jit_im,true_shift = tls.im_to_jitter(im,tls.jit_speed,true_I,memory_range)
    I = 15
    subpixel_nb = 10
    dejit_im,shifts,t = main_DP_on_the_grid_L1_regul2(jit_im[:,:-int(np.ceil(true_I))], I, subpixel_nb, gamma, "demo_DP_"+name, True, memory_range)
    tls.plot(dejit_im, "Dejittered image using DP algorithm")
    tls.plot(jit_im[-128:,100+I:228+I], "Crop of the jittered image")
    tls.plot(im[-128:,100+I:228+I], "Crop of the original image")
    tls.plot(dejit_im[-128:,100:228], "Crop of the dejittered image using DP algorithm")
    plt.figure()
    plt.plot(true_shift[I:-I])
    plt.plot(shifts)
    plt.show()
    print("PSNR is ",peak_signal_noise_ratio(im[:,I:-I-int(np.ceil(true_I))],dejit_im,data_range = 2**memory_range-1))
    print("compared to ",peak_signal_noise_ratio(im[:,I:-I-int(np.ceil(true_I))],jit_im[:,I:-I-int(np.ceil(true_I))],data_range = 2**memory_range-1))
    print("SSIM is ",structural_similarity(im[:,I:-I-int(np.ceil(true_I))],dejit_im,data_range = 2**memory_range-1))
    print("compared to ",structural_similarity(im[:,I:-I-int(np.ceil(true_I))],jit_im[:,I:-I-int(np.ceil(true_I))],data_range = 2**memory_range-1))
