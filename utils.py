#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:06:05 2020

@author: landry@innopsys.lan
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(Im,title=None):
    """
    plot image Im in shade of gray without axes, with optional title.

    Parameters
    ----------
    Im : np.array (2D) of shape (N_2,N_1)
        image to be plotted.
    title : string, optional
        title of the plot. The default is None.

    Returns
    -------
    None.

    """
    fig = plt.figure()
    fig = plt.imshow(Im,cmap='gray')
    plt.axis=('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if type(title) != type(None):
        plt.title(title)
    plt.show()
    
def im_to_jitter(im,jit_fun,arg,memory_range = 8):
    """
    Performs in silico jitter of image im with jittering function jit_fun.

    Parameters
    ----------
    im : np.array (2D) of shape (N_2,N_1)
        image to be jittered.
    jit_fun : python function
        the image is jittered under jit_fun jitter function.
    arg : float
        argument to be passed in jit_fun.
    memory_range : integer, optional
        number of bits for a pixel. The default is 8.

    Returns
    -------
    new_im : np.array (2D) of shape (N_2,N_1)
        jittered image clipped between 0,2**8-1. in np.uint8 or 16 depending on memory_range.
    jit : list of float of length (N_1)
        list of shifts used at each pixel.

    """
    new_im = np.copy(im)
    jit = []
    for j in range(im.shape[1]):
        new_im[1::2,j],tmp = jit_fun(j,im[1::2,:],arg,memory_range)
        jit.append(tmp)
    if memory_range <= 8:
        return np.clip(new_im,0,2**memory_range-1).astype(np.uint8),jit
    else:
        return np.clip(new_im,0,2**memory_range-1).astype(np.uint16),jit

    
def jit_speed(j,odd_data,arg,memory_range,min_pos=-0.9,max_pos=0.9):
    """
    Computes the column of the in silico jittered image.
    The jitter function is of the kind observed by Innopsys.
    

    Parameters
    ----------
    j : integer
        current pixel position.
    odd_data : np.array of shape (n_2,N_1)
        odd lines of the image to be jittered.
    arg : float
        maximum jitter (in pixel).
    memory_range : integer
        number of bits for a pixel.
    min_pos : float \in [-1,1]
        x position of the start of the image.
    max_pos : float \in [-1,1]
        x position of the end of the image (0 is the middle).
    
    Returns
    -------
    TYPE
        value of current jittered pixel.
    vitesse : float
        shift used for jittering.

    """
    vitesse = arg*np.sqrt(1-(j/odd_data.shape[1]*(max_pos-min_pos)+min_pos)**2)
    vitesse_int_fl = int(np.floor(vitesse))
    vitesse_int_ce = int(np.ceil(vitesse))
    vitesse_float = vitesse-vitesse_int_fl
    if vitesse_int_ce+j < odd_data.shape[1] and vitesse_int_fl+j >= 0:
        # linear interpolation
        return odd_data[:,j+vitesse_int_fl]*(1-vitesse_float) + odd_data[:,j+vitesse_int_ce]*vitesse_float , vitesse
    else:
        # fill value when out of bounds
        return 2**memory_range-1,vitesse
    
def jit_const(j,odd_data,arg,memory_range):
    """
    Computes the column of the in silico jittered image.
    The jitter function is constant.

    Parameters
    ----------
    j : integer
        current pixel position.
    odd_data : np.array of shape (n_2,N_1)
        odd lines of the image to be jittered.
    arg : float
        maximum jitter (in pixel).
    memory_range : integer
        number of bits for a pixel.
    min_pos : float \in [-1,1]
        x position of the start of the image.
    max_pos : float \in [-1,1]
        x position of the end of the image (0 is the middle).
    
    Returns
    -------
    TYPE
        value of current jittered pixel.
    vitesse : float
        shift used for jittering.

    """
    shift = arg
    vitesse_int_fl = int(np.floor(shift))
    vitesse_int_ce = int(np.ceil(shift))
    vitesse_float = shift-vitesse_int_fl
    if vitesse_int_ce+j < odd_data.shape[1] and vitesse_int_fl+j >= 0:
        return odd_data[:,j+vitesse_int_fl]*(1-vitesse_float) + odd_data[:,j+vitesse_int_ce]*vitesse_float , shift
    else:
        return 2**memory_range-1,shift

def jit_composed(j,odd_data,arg,memory_range):
    """
    Computes the column of the in silico jittered image.
    The jitter function is composed of a linear and a sinusoidal part.

    Parameters
    ----------
    j : integer
        current pixel position.
    odd_data : np.array of shape (n_2,N_1)
        odd lines of the image to be jittered.
    arg : float
        maximum jitter (in pixel).
    memory_range : integer
        number of bits for a pixel.
    min_pos : float \in [-1,1]
        x position of the start of the image.
    max_pos : float \in [-1,1]
        x position of the end of the image (0 is the middle).
    
    Returns
    -------
    TYPE
        value of current jittered pixel.
    vitesse : float
        shift used for jittering.

    """
    n = odd_data.shape[1]
    
    vitesse = -arg*(-j/n*(j<0.5*n) + 0.4*np.sin(j*(j>=0.5*n)*2*np.pi/n)-0.5*(j>=0.5*n))
    vitesse_int_fl = int(np.floor(vitesse))
    vitesse_int_ce = int(np.ceil(vitesse))
    vitesse_float = vitesse-vitesse_int_fl
    if vitesse_int_ce+j < odd_data.shape[1] and vitesse_int_fl+j >= 0:
        return odd_data[:,j+vitesse_int_fl]*(1-vitesse_float) + odd_data[:,j+vitesse_int_ce]*vitesse_float , vitesse
    else:
        return 2**memory_range-1,vitesse
