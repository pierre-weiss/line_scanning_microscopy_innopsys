#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:24:06 2020

@author: landry@innopsys.lan
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from time import time
import utils as tls
from numba import jit

def fast_recons_im(Odd,w_sol,I,memory_range,opt = 0):
    """
    Applies the weights w_sol to image Odd in order to recover the original image

    Parameters
    ----------
    Odd : np.array
        odd lines of the image.
    w_sol : np.array
        weight matrix for dejittering.
    I : integer
        search pattern size.
    memory_range : int
        how many bits for a pixel in the image.
    opt : boolean, optional
        help handling with w shape (for different 1D or 2D outputs). The default is 0.

    Returns
    -------
    np.array
        reconstructed odd lines based on matrix W.

    """
    n = Odd.shape[1]-I
    if opt:
        w = np.reshape(w_sol,(-1,I)).T
    else:
        w = w_sol
        w_sol = np.reshape(w.T,(-1))
    # tls.plot(w)
    # plt.plot(np.sum(w,axis = 0))
    new_Odd = np.zeros((Odd.shape[0],Odd.shape[1]))
    n_1,n_2 = Odd.shape
    for i in range(n):
        new_Odd[:,I+i] = Odd[:,i:i+I]@w[:,i]
    if memory_range<=8:        
        return np.clip(new_Odd,0,2**memory_range-1).astype(np.uint8)
    else:
        return np.clip(new_Odd,0,2**memory_range-1).astype(np.uint16)

#%% proximal weight relaxation and projection to the simplex algorithms
def solve_prox_DR(im,I,gamma,W0,show = 0):
    """
    Find the weights solving the proximal operator needed in the Douglas-Rachford iteration.
    (the proximal operator is constructed and solved using a DP algorithm)

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern size.
    gamma : float
        regularization parameter.
    W0 : np.array
        Where to evaluate the proximal operator.
    show : boolean, optional
        if True, display messages during the iterations. The default is 0.

    Returns
    -------
    np.array
    Weights solution of the proximal operator

    """
    ### W0 : prox pour DR évalué en W0.
    t0 = time()
    Even,Odd = im[::2,:].astype(np.float64),im[1::2,:].astype(np.float64)
    n_1,n_2 = Even.shape
    list_hat_A_i = []
    list_hat_B_i = []
    A_i = np.zeros((I,I))
    B_i = np.zeros((I,1))
    ### Initialisation des hat_A_i et hat_B_i
    A_i = Odd[:-1,:I].T.dot(Odd[:-1,:I])*4/(n_1-1)+1/2*np.eye(I) ### DR : +1/2 I
    B_i = np.expand_dims((Even[:-1,I]+Even[1:,I])@Odd[:-1,:I]*(-2)/(n_1-1)- W0[:I],axis=1)  ### DR: +1/2 (-2*W0)
    list_hat_A_i.append(A_i)
    list_hat_B_i.append(B_i)
    for i in np.arange(I+1,n_2): ### boucle sur les noeuds de la chaine
        ### construction A_(i-1,i) et B_(i-1,i) du message de l'arête courante
        if i%3000 == 0 and show:
            print("    Constructing messages at column : ",i," - time : ",time()-t0," - ",int(i/n_2*100),"% of processed columns")
        tmp_mat = np.linalg.inv(list_hat_A_i[-1]/gamma+np.eye(I))
        tmp_mat2 = tmp_mat@list_hat_A_i[-1]@tmp_mat
        tmp_mat_carr = tmp_mat@tmp_mat
        A_im1i = np.zeros((I,I))
        B_im1i = np.zeros((I,1))
        A_im1i = tmp_mat2+gamma*np.eye(I)-2*gamma*tmp_mat+gamma*tmp_mat_carr
        B_im1i = tmp_mat2@(-list_hat_B_i[-1]/gamma)+2*tmp_mat@list_hat_B_i[-1]-tmp_mat_carr@list_hat_B_i[-1]
        ### construction A_i et B_i de la fonction du noeud courant
        A_i = np.zeros((I,I))
        B_i = np.zeros((I,1))
        A_i = Odd[:-1,i-I:i].T.dot(Odd[:-1,i-I:i])*4/(n_1-1)+1/2*np.eye(I) ### DR : +1/2 I
        B_i = np.expand_dims((Even[:-1,i]+Even[1:,i])@Odd[:-1,i-I:i]*(-2)/(n_1-1)- W0[(i-I)*I:(i+1-I)*I],axis=1) ### DR: +1/2 (-2*W0)
        ### construction hat_A_i et hat_B_i du message du noeud courant
        list_hat_A_i.append(np.copy(A_i+A_im1i))
        list_hat_B_i.append(np.copy(B_i+B_im1i))
    ### resolution pour le noeud racine
    sol_w = []
    sol_w.append(np.linalg.solve(list_hat_A_i[-1], -list_hat_B_i[-1]))
    for i in np.arange(n_2-2,I-1,-1): ### backward propagation
        sol_w.append(np.linalg.solve(list_hat_A_i[i-I]/gamma+np.eye(I),sol_w[-1]-list_hat_B_i[i-I]/gamma))
    sol_w = sol_w[::-1]
    ### return les W_i sol
    if show:
        print("    Resolution time : ",time()-t0)
    return np.reshape(np.array(sol_w)[:,:,0],(-1))

@jit
def pre_computer_prox_DR(im,I,gamma,show = 0):
    """
    Find the weights solving the proximal operator needed in the Douglas-Rachford iteration.
    (the proximal operator is constructed and solved using a DP algorithm)

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern size.
    gamma : float
        regularization parameter.
    show : boolean, optional
        if True, display messages during the iterations. The default is 0.

    Returns
    -------
    np.array
    Weights solution of the proximal operator

    """
    ### W0 : prox pour DR évalué en W0.
    t0 = time()
    Even,Odd = im[::2,:].astype(np.float64),im[1::2,:].astype(np.float64)
    n_1,n_2 = Even.shape
    list_hat_A_i = []
    list_B_i = []
    A_i = np.zeros((I,I))
    B_i = np.zeros(I)
    ### Initialisation des hat_A_i et hat_B_i
    A_i = Odd[:-1,:I].T.dot(Odd[:-1,:I])*4/(n_1-1)+1/2*np.eye(I) ### DR : +1/2 I
    B_i = (Even[:-1,I]+Even[1:,I])@Odd[:-1,:I]*(-2)/(n_1-1)  ### DR: +1/2 (-2*W0)
    list_hat_A_i.append(A_i)
    list_B_i.append(B_i)
    list_tmp_mat = []
    list_tmp_mat2 = []
    list_tmp_mat_carr = []
    for i in np.arange(I+1,n_2): ### boucle sur les noeuds de la chaine
        ### construction A_(i-1,i) et B_(i-1,i) du message de l'arête courante
        if i%3000 == 0 and show:
            print("    Pre-Constructing messages at column : ",i," - time : ",time()-t0," - ",int(i/n_2*100),"% of processed columns")
        # tmp_mat = np.linalg.inv(list_hat_A_i[-1]/gamma+np.eye(I))
        list_tmp_mat.append(np.linalg.inv(list_hat_A_i[-1]/gamma+np.eye(I)))
        # tmp_mat2 = tmp_mat@list_hat_A_i[-1]@tmp_mat
        list_tmp_mat2.append(list_tmp_mat[-1]@list_hat_A_i[-1]@list_tmp_mat[-1])
        # tmp_mat_carr = tmp_mat@tmp_mat
        list_tmp_mat_carr.append(list_tmp_mat[-1]@list_tmp_mat[-1])
        A_im1i = np.zeros((I,I))
        A_im1i = list_tmp_mat2[-1]+gamma*np.eye(I)-2*gamma*list_tmp_mat[-1]+gamma*list_tmp_mat_carr[-1]
        ### construction A_i et B_i de la fonction du noeud courant
        A_i = np.zeros((I,I))
        B_i = np.zeros(I)
        A_i = Odd[:-1,i-I:i].T.dot(Odd[:-1,i-I:i])*4/(n_1-1)+1/2*np.eye(I) ### DR : +1/2 I
        B_i = (Even[:-1,i]+Even[1:,i])@Odd[:-1,i-I:i]*(-2)/(n_1-1) ### DR: +1/2 (-2*W0)
        ### construction hat_A_i et hat_B_i du message du noeud courant
        list_hat_A_i.append(np.copy(A_i+A_im1i))
        list_B_i.append(np.copy(B_i))
    if show:
        print("    Total pre-computing time : ",time()-t0)
    return list_hat_A_i, list_B_i, list_tmp_mat, list_tmp_mat2, list_tmp_mat_carr

@jit
def solve_prox_DR_from_pre_computed_data(im,I,gamma,W0,list_hat_A_i,list_B,list_mat,list_mat2,list_mat_square):
    """
    Find the weights solving the proximal operator needed in the Douglas-Rachford iteration.
    (the proximal operator is constructed and solved using a DP algorithm)

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern size.
    gamma : float
        regularization parameter.
    W0 : np.array
        Where to evaluate the proximal operator.
    show : boolean, optional
        if True, display messages during the iterations. The default is 0.

    Returns
    -------
    np.array
    Weights solution of the proximal operator

    """
    ### W0 : prox pour DR évalué en W0.
    n_1,n_2 = im.shape
    n_1 = n_1//2
    # list_hat_A_i = []
    list_hat_B_i = []
    # A_i = np.zeros((I,I))
    # B_i = np.zeros((I,1))
    ### Initialisation des hat_A_i et hat_B_i
    # A_i = Odd[:-1,:I].T.dot(Odd[:-1,:I])*4/(n_1-1)+1/2*np.eye(I) ### DR : +1/2 I
    # B_i = np.expand_dims((Even[:-1,I]+Even[1:,I])@Odd[:-1,:I]*(-2)/(n_1-1)- W0[:I],axis=1)  ### DR: +1/2 (-2*W0)
    # list_hat_A_i.append(list_A[0])
    list_hat_B_i.append(np.expand_dims(list_B[0]- W0[:I],axis=1))
    for it,i in enumerate(np.arange(I+1,n_2)): ### boucle sur les noeuds de la chaine
        ### construction A_(i-1,i) et B_(i-1,i) du message de l'arête courante
        # A_im1i = np.zeros((I,I))
        B_im1i = np.zeros((I,1))
        # A_im1i = tmp_mat2+gamma*np.eye(I)-2*gamma*tmp_mat+gamma*tmp_mat_carr
        B_im1i = list_mat2[it]@(-list_hat_B_i[-1]/gamma)+2*list_mat[it]@list_hat_B_i[-1]-list_mat_square[it]@list_hat_B_i[-1]
        ### construction A_i et B_i de la fonction du noeud courant
        # A_i = np.zeros((I,I))
        B_i = np.zeros((I,1))
        # A_i = Odd[:-1,i-I:i].T.dot(Odd[:-1,i-I:i])*4/(n_1-1)+1/2*np.eye(I) ### DR : +1/2 I
        B_i = np.expand_dims(list_B[it+1]- W0[(i-I)*I:(i+1-I)*I],axis=1) ### DR: +1/2 (-2*W0)
        ### construction hat_A_i et hat_B_i du message du noeud courant
        # list_hat_A_i.append(np.copy(A_i+A_im1i))
        list_hat_B_i.append(np.copy(B_i+B_im1i))
    ### resolution pour le noeud racine
    sol_w = []
    sol_w.append(np.linalg.solve(list_hat_A_i[-1], -list_hat_B_i[-1]))
    for i in np.arange(n_2-2,I-1,-1): ### backward propagation
        sol_w.append(np.linalg.solve(list_hat_A_i[i-I]/gamma+np.eye(I),sol_w[-1]-list_hat_B_i[i-I]/gamma))
    sol_w = sol_w[::-1]
    ### return les W_i sol
    return np.reshape(np.array(sol_w)[:,:,0],(-1))

def proj_comp_simplex(v,I,n):
    """
    projects slices of the weights on the simplex

    Parameters
    ----------
    v : np.array
        vector of the weights.
    I : integer
        search pattern size.
    n : integer
        length of the jittered image (2nd dim).

    Returns
    -------
    p : np.array
        projected weight vector on the simplex.

    """
    p = np.zeros(v.shape)
    for i in range(n):
        p[i*I:(i+1)*I] = projection_simplex_sort(v[i*I:(i+1)*I])
    return p

def projection_simplex_sort(v, z=1):
    """
    Projects vector v to the simplex defined by the sets of positive vectors whose 1-norm is 1

    Parameters
    ----------
    v : np.array
        vector to be projected.
    z : float, optional
        defines the 1-norm of the resulting projection. The default is 1.

    Returns
    -------
    w : np.array
        projected vector.

    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def solver_DR(im,I,gamma,tol = 1e-3, show = False):
    """
    solves the relaxed weights problem using douglas rachford algorithm

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern size.
    gamma : float
        regularization parameter.
    tol : float, optional
        tolerance of the douglas rachford algorithm. The default is 1e-3.
    show : boolean, optional
        whether of not to display informations on the solver (recommended only for big images). The default is False.

    Returns
    -------
    x_new : np.array
        solution of the relaxed weights problem
    it : integer
        number of douglas rachford iterations

    """
    t0 = time()
    n = im.shape[1]-I
    y_old = np.zeros(n*I)
    x_old = np.zeros(n*I)
    x_new = np.copy(proj_comp_simplex(y_old,I,n))
    y_new = np.copy(solve_prox_DR(im, I, gamma, 2*x_new, show)-x_new)
    it = 1
    while np.linalg.norm(x_new-x_old)>tol*np.linalg.norm(x_new):
        it+=1
        y_old = y_new
        x_old = x_new
        x_new = proj_comp_simplex(y_old,I,n)
        y_new = y_old + solve_prox_DR(im, I, gamma, 2*x_new-y_old, show)-x_new
        if show and it%10==0:
            print("Ongoing Douglas Rachford total time: ",time()-t0," - actual precision = ",np.linalg.norm(x_new-x_old)/np.linalg.norm(x_new)," for tolerance = ",tol)
    return x_new,it

@jit
def pre_computed_solver_DR(im,I,gamma,tol = 1e-3, show = False):
    """
    solves the relaxed weights problem using douglas rachford algorithm

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern size.
    gamma : float
        regularization parameter.
    tol : float, optional
        tolerance of the douglas rachford algorithm. The default is 1e-3.
    show : boolean, optional
        whether of not to display informations on the solver (recommended only for big images). The default is False.

    Returns
    -------
    x_new : np.array
        solution of the relaxed weights problem
    it : integer
        number of douglas rachford iterations

    """
    t0 = time()
    n = im.shape[1]-I
    l1, l2, l3, l4, l5 = pre_computer_prox_DR(im, I, gamma, show)
    y_old = np.zeros(n*I)
    x_old = np.zeros(n*I)
    x_new = np.copy(proj_comp_simplex(y_old,I,n))
    y_new = np.copy(solve_prox_DR_from_pre_computed_data(im, I, gamma, 2*x_new, l1, l2, l3, l4, l5)-x_new)
    it = 1
    while np.linalg.norm(x_new-x_old)>tol*np.linalg.norm(x_new):
        it+=1
        y_old = y_new
        x_old = x_new
        x_new = proj_comp_simplex(y_old,I,n)
        y_new = y_old + solve_prox_DR_from_pre_computed_data(im, I, gamma, 2*x_new-y_old, l1, l2, l3, l4, l5)-x_new
        if show and it%10==0:
            print("Ongoing Douglas Rachford total time: ",time()-t0," - actual precision = ",np.linalg.norm(x_new-x_old)/np.linalg.norm(x_new)," for tolerance = ",tol)
    return x_new,it

#%% main algorithms constrained weight relaxation
def main_solver_weight_relax(im,I,mu,name="",save=False,memory_range = 16, show = False):
    """
    

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern length.
    mu : float
        regularization parameter.
    name : string, optional
        name of the image when saved. The default is "".
    save : boolean, optional
        whether or not to save the image. The default is False.
    memory_range : integer, optional
        number of bits on which the image is written. The default is 16.
    show : boolean, optional
        whether or not to display solver informations. The default is False.

    Returns
    -------
    np.array
        weights solution of the weight relaxation problem.
    np.array
        dejittered image.
    float
        time taken to solve the problem and reconstruct the image.

    """
    if show:
        print("Starting the Weight Relaxation algorithm")
    t0 = time()
    sol_x,it = solver_DR(im, I, mu, show = show)
    t2 = time()
    dej_im = np.copy(im)
    if show:
        print("Number of iteration for Douglas Rachford : ",it)
        print("Applying the weights on the jittered image")
    dej_im[1::2,:] = fast_recons_im(im[1::2,:], sol_x, I, memory_range,1)
    t1 = time()
    if show:
        print("Image reconstructed in ",t1-t2)
    if save:
        if memory_range<=8:
            io.imsave("../images/dejitt_Weight_Relax_"+name+".png",np.clip(dej_im[:,I:-I],0,2**memory_range-1).astype(np.uint8))
        else:
            io.imsave("../images/dejitt_Weight_Relax_"+name+".png",np.clip(dej_im[:,I:-I],0,2**memory_range-1).astype(np.uint16))
        if show:
            t2 = time()
            print("Reconstructed image saved in ",t2-t1)
    t1 = time()
    if show:
        print("Total time === ",t1-t0)
    return sol_x,dej_im[:,I:-I],t1-t0

def demo_weight_relax(opt = 0):
    """
    Demo of main_solver_weight_relax use for 2 possible cases (Chelsea and V_manu)
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
        mu = 30000
    else:
        im = np.array(io.imread("../images/V_manu.png")).astype(np.float64)
        name = "V_manu"
        mu = 200000
    memory_range = 8
    true_I = 10.4
    jit_im,true_shift = tls.im_to_jitter(im,tls.jit_speed,true_I,memory_range)
    I = 15
    sol,dejit_im,t = main_solver_weight_relax(jit_im[:,:-int(np.ceil(true_I))], I, mu, "demo_WRelax_"+name, True, memory_range, True)
    tls.plot(np.reshape(sol,(-1,I)).T, "Weight matrix solution of the weight relaxation algorithm")
    tls.plot(dejit_im, "Dejittered image using weight relaxation algorithm")
    tls.plot(jit_im[-128:,100+I:228+I], "Crop of the jittered image")
    tls.plot(im[-128:,100+I:228+I], "Crop of the original image")
    tls.plot(dejit_im[-128:,100:228], "Crop of the dejittered image using weight relaxation algorithm")
    print("PSNR is ",peak_signal_noise_ratio(im[:,I:-I-int(np.ceil(true_I))],dejit_im,data_range = 2**memory_range-1))
    print("compared to ",peak_signal_noise_ratio(im[:,I:-I-int(np.ceil(true_I))],jit_im[:,I:-I-int(np.ceil(true_I))],data_range = 2**memory_range-1))
    print("SSIM is ",structural_similarity(im[:,I:-I-int(np.ceil(true_I))],dejit_im,data_range = 2**memory_range-1))
    print("compared to ",structural_similarity(im[:,I:-I-int(np.ceil(true_I))],jit_im[:,I:-I-int(np.ceil(true_I))],data_range = 2**memory_range-1))



#%% main algorithms constrained weight relaxation 2
def main_solver_weight_relax_with_pre_computer(im,I,mu,tol=1e-3,name="",save=False,memory_range = 16, show = False):
    """
    

    Parameters
    ----------
    im : np.array
        jittered image.
    I : integer
        search pattern length.
    mu : float
        regularization parameter.
    tol : float, optional
        tolerance of the douglas rachford algorithm. The default is 1e-3.
    name : string, optional
        name of the image when saved. The default is "".
    save : boolean, optional
        whether or not to save the image. The default is False.
    memory_range : integer, optional
        number of bits on which the image is written. The default is 16.
    show : boolean, optional
        whether or not to display solver informations. The default is False.

    Returns
    -------
    np.array
        weights solution of the weight relaxation problem.
    np.array
        dejittered image.
    float
        time taken to solve the problem and reconstruct the image.

    """
    if show:
        print("Starting the Weight Relaxation algorithm")
    t0 = time()
    sol_x,it = pre_computed_solver_DR(im, I, mu, tol=tol, show = show)
    t2 = time()
    dej_im = np.copy(im)
    if show:
        print("Number of iteration for Douglas Rachford : ",it)
        print("Applying the weights on the jittered image")
    dej_im[1::2,:] = fast_recons_im(im[1::2,:], sol_x, I, memory_range,1)
    t1 = time()
    if show:
        print("Image reconstructed in ",t1-t2)
    if save:
        if memory_range<=8:
            io.imsave("../images/dejitt_Weight_Relax_"+name+".png",np.clip(dej_im[:,I:-I],0,2**memory_range-1).astype(np.uint8))
        else:
            io.imsave("../images/dejitt_Weight_Relax_"+name+".png",np.clip(dej_im[:,I:-I],0,2**memory_range-1).astype(np.uint16))
        if show:
            t2 = time()
            print("Reconstructed image saved in ",t2-t1)
    t1 = time()
    if show:
        print("Total time === ",t1-t0)
    return sol_x,dej_im[:,I:-I],t1-t0

def demo_weight_relax_with_pre_computer(opt = 0):
    """
    Demo of main_solver_weight_relax use for 2 possible cases (Chelsea and V_manu)
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
        mu = 30000
    else:
        im = np.array(io.imread("../images/V_manu.png")).astype(np.float64)
        name = "V_manu"
        mu = 200000
    memory_range = 8
    true_I = 10.4
    jit_im,true_shift = tls.im_to_jitter(im,tls.jit_speed,true_I,memory_range)
    I = 15
    sol,dejit_im,t = main_solver_weight_relax_with_pre_computer(jit_im[:,:-int(np.ceil(true_I))], I, mu, "demo_WRelax_"+name, True, memory_range, True)
    tls.plot(np.reshape(sol,(-1,I)).T, "Weight matrix solution of the weight relaxation algorithm")
    tls.plot(dejit_im, "Dejittered image using weight relaxation algorithm")
    tls.plot(jit_im[-128:,100+I:228+I], "Crop of the jittered image")
    tls.plot(im[-128:,100+I:228+I], "Crop of the original image")
    tls.plot(dejit_im[-128:,100:228], "Crop of the dejittered image using weight relaxation algorithm")
    print("PSNR is ",peak_signal_noise_ratio(im[:,I:-I-int(np.ceil(true_I))],dejit_im,data_range = 2**memory_range-1))
    print("compared to ",peak_signal_noise_ratio(im[:,I:-I-int(np.ceil(true_I))],jit_im[:,I:-I-int(np.ceil(true_I))],data_range = 2**memory_range-1))
    print("SSIM is ",structural_similarity(im[:,I:-I-int(np.ceil(true_I))],dejit_im,data_range = 2**memory_range-1))
    print("compared to ",structural_similarity(im[:,I:-I-int(np.ceil(true_I))],jit_im[:,I:-I-int(np.ceil(true_I))],data_range = 2**memory_range-1))

#%% tools

def shift_from_w(w,X):
    """
    recover the shifts from the solution weights of the weight relaxation problem:
    the j-th shift is the argmax of the j-th weigths column interpolated on a thin grid.
    (serves only as an indicator of performance)

    Parameters
    ----------
    w : np.array
        solution weights to be transformed in shifts.
    X : integer
        grid refinement term: there is X subpixels in 1 pixel.

    Returns
    -------
    np.array
        shifts obtained as argmax of the interpolated weights.

    """
    search_pattern_length = w.shape[1]
    tmp = np.zeros((w.shape[0],search_pattern_length*X-X))
    for i in range(X*(search_pattern_length-1)):
        tmp[:,i] = w[:,int(np.floor(i/X))]*(1-i/X+int(np.floor(i/X)))+w[:,int(np.floor(i/X))]*(i/X-int(np.floor(i/X)))
    d = np.argmax(tmp,axis = 1)
    return -(d/X-search_pattern_length)[:-search_pattern_length]




