#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:38:30 2020

@author: landry@innopsys.lan
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import utils as tls
import DP_algo as DP
import Weight_Relaxation_algo as W_Relax


im1 = np.array(io.imread("Images/Chelsea.png")).astype(np.float64)*2**8
name1 = "Chelsea"
im2 = np.array(io.imread("Images/V_manu.png")).astype(np.float64)*2**8
name2 = "V_manu"

true_I = 10.4
search_zone_length = 12
subpixels_number = 10

memory_range = 16
jit_im1,true_shift1 = tls.im_to_jitter(im1,tls.jit_const,true_I,memory_range)
jit_im2,true_shift2 = tls.im_to_jitter(im1,tls.jit_speed,true_I,memory_range)
jit_im3,true_shift3 = tls.im_to_jitter(im1,tls.jit_composed,true_I,memory_range)
jit_im4,true_shift4 = tls.im_to_jitter(im2,tls.jit_const,true_I,memory_range)
jit_im5,true_shift5 = tls.im_to_jitter(im2,tls.jit_speed,true_I,memory_range)
jit_im6,true_shift6 = tls.im_to_jitter(im2,tls.jit_composed,true_I,memory_range)

true_shift1 = true_shift1[:-int(true_I)-1]
true_shift2 = true_shift2[:-int(true_I)]
true_shift3 = true_shift3[:-int(true_I)]
jit_im1 = jit_im1[:,:-int(true_I)-1]
jit_im2 = jit_im2[:,:-int(true_I)]
jit_im3 = jit_im3[:,:-int(true_I)]

io.imsave("jit_Chelsea_const.png",(jit_im1[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("jit_Chelsea_speed.png",(jit_im2[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("jit_Chelsea_composed.png",(jit_im3[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("jit_V_manu_const.png",(jit_im4[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("jit_V_manu_speed.png",(jit_im5[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("jit_V_manu_composed.png",(jit_im6[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))

algo1 = "DP"
algo2 = "Relax"

dej1,d1,t1 = DP.main_DP_on_the_grid_L1_regul2(jit_im1,search_zone_length,subpixels_number,50000)
dej2,d2,t2 = DP.main_DP_on_the_grid_L1_regul2(jit_im2,search_zone_length,subpixels_number,15000)
dej3,d3,t3 = DP.main_DP_on_the_grid_L1_regul2(jit_im3,search_zone_length,subpixels_number,15000)
dej4,d4,t4 = DP.main_DP_on_the_grid_L1_regul2(jit_im4,search_zone_length,subpixels_number,700)
dej5,d5,t5 = DP.main_DP_on_the_grid_L1_regul2(jit_im5,search_zone_length,subpixels_number,2000)
dej6,d6,t6 = DP.main_DP_on_the_grid_L1_regul2(jit_im6,search_zone_length,subpixels_number,2500)

d1dr,dej1dr,t1dr = W_Relax.main_solver_weight_relax_with_pre_computer(jit_im1,search_zone_length,5e5,tol=1e-4)
d2dr,dej2dr,t2dr = W_Relax.main_solver_weight_relax_with_pre_computer(jit_im2,search_zone_length,5e5,tol=1e-4)
d3dr,dej3dr,t3dr = W_Relax.main_solver_weight_relax_with_pre_computer(jit_im3,search_zone_length,2e6,tol=1e-4)
d4dr,dej4dr,t4dr = W_Relax.main_solver_weight_relax_with_pre_computer(jit_im4,search_zone_length,1e8,tol=1e-4)
d5dr,dej5dr,t5dr = W_Relax.main_solver_weight_relax_with_pre_computer(jit_im5,search_zone_length,1e8,tol=1e-4)
d6dr,dej6dr,t6dr = W_Relax.main_solver_weight_relax_with_pre_computer(jit_im6,search_zone_length,6e6,tol=1e-4)

d1dr = W_Relax.shift_from_w(np.reshape(d1dr,(-1,search_zone_length)),100)
d2dr = W_Relax.shift_from_w(np.reshape(d2dr,(-1,search_zone_length)),100)
d3dr = W_Relax.shift_from_w(np.reshape(d3dr,(-1,search_zone_length)),100)
d4dr = W_Relax.shift_from_w(np.reshape(d4dr,(-1,search_zone_length)),100)
d5dr = W_Relax.shift_from_w(np.reshape(d5dr,(-1,search_zone_length)),100)
d6dr = W_Relax.shift_from_w(np.reshape(d6dr,(-1,search_zone_length)),100)

im1 = im1[:,search_zone_length:-search_zone_length]
im2 = im2[:,search_zone_length:-search_zone_length]

#%% plotting the different found shifts compared to the real one
fig, axs = plt.subplots(3,2)
axs[0,0].plot(np.linspace(0,1,len(true_shift1)-2*search_zone_length),true_shift1[12:-12],linewidth=1)
axs[0,1].plot(np.linspace(0,1,len(true_shift4)-2*search_zone_length),true_shift4[12:-12],linewidth=1)
axs[0,0].plot(np.linspace(0,1,len(true_shift1)-2*search_zone_length),d1dr,'--',linewidth=1)
axs[0,0].plot(np.linspace(0,1,len(true_shift1)-2*search_zone_length),d1,'-.',linewidth=1)
axs[0,1].plot(np.linspace(0,1,len(true_shift4)-2*search_zone_length),d4dr,'--',linewidth=1)
axs[0,1].plot(np.linspace(0,1,len(true_shift4)-2*search_zone_length),d4,'-.',linewidth=1)
axs[0,0].legend(["true shift","Relax "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift1[12:-12]-d1dr)),"DP "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift1[12:-12]-d1))],fontsize='x-small')
axs[0,1].legend(["true shift","Relax "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift4[12:-12]-d4dr)),"DP "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift4[12:-12]-d4))],fontsize='x-small')

axs[1,0].plot(np.linspace(0,1,len(true_shift2)-2*search_zone_length),true_shift2[12:-12],linewidth=1)
axs[1,1].plot(np.linspace(0,1,len(true_shift5)-2*search_zone_length),true_shift5[12:-12],linewidth=1)
axs[1,0].plot(np.linspace(0,1,len(true_shift2)-2*search_zone_length),d2dr,'--',linewidth=1)
axs[1,0].plot(np.linspace(0,1,len(true_shift2)-2*search_zone_length),d2,'-.',linewidth=1)
axs[1,1].plot(np.linspace(0,1,len(true_shift5)-2*search_zone_length),d5dr,'--',linewidth=1)
axs[1,1].plot(np.linspace(0,1,len(true_shift5)-2*search_zone_length),d5,'-.',linewidth=1)
axs[1,0].legend(["true shift","Relax "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift2[12:-12]-d2dr)),"DP "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift2[12:-12]-d2))],fontsize='x-small')
axs[1,1].legend(["true shift","Relax "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift5[12:-12]-d5dr)),"DP "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift5[12:-12]-d5))],fontsize='x-small')

axs[2,0].plot(np.linspace(0,1,len(true_shift3)-2*search_zone_length),true_shift3[12:-12],linewidth=1)
axs[2,1].plot(np.linspace(0,1,len(true_shift6)-2*search_zone_length),true_shift6[12:-12],linewidth=1)
axs[2,0].plot(np.linspace(0,1,len(true_shift3)-2*search_zone_length),d3dr,'--',linewidth=1)
axs[2,0].plot(np.linspace(0,1,len(true_shift3)-2*search_zone_length),d3,'-.',linewidth=1)
axs[2,1].plot(np.linspace(0,1,len(true_shift6)-2*search_zone_length),d6dr,'--',linewidth=1)
axs[2,1].plot(np.linspace(0,1,len(true_shift6)-2*search_zone_length),d6,'-.',linewidth=1)
axs[2,0].legend(["true shift","Relax "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift3[12:-12]-d3dr)),"DP "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift3[12:-12]-d3))],fontsize='x-small')
axs[2,1].legend(["true shift","Relax "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift6[12:-12]-d6dr)),"DP "+"$q_{0.5}^{abs}$ = %5.3f" % np.median(np.abs(true_shift6[12:-12]-d6))],fontsize='x-small')


i=0
x_label = ["Cat Image","V Image"]
y_label = ["Constant jitter","Innopsys jitter","Composed jitter"]
for ax in axs.flat:
    if i%2:
        ax.set(xlabel=x_label[i%2])
    else:
        ax.set(xlabel=x_label[i%2], ylabel=y_label[int((i+1)/2)])
    i+=1

plt.savefig("Comparison_all_shifts.png")

#%% saving resulting images and computing scores for each algorithms (PSNR, SSIM, median of the absolute shift error, and time)

io.imsave("dej_"+algo1+"_Chelsea_const.png",(dej1/256).astype(np.uint8))
io.imsave("dej_"+algo1+"_Chelsea_speed.png",(dej2/256).astype(np.uint8))
io.imsave("dej_"+algo1+"_Chelsea_composed.png",(dej3/256).astype(np.uint8))
io.imsave("dej_"+algo1+"_V_manu_const.png",(((dej4-np.min(dej4))/(np.max(dej4)-np.min(dej4))*255).astype(np.uint8))) #the contrast displaying norm in latex makes artefacts in the text image so we renormalize its values
io.imsave("dej_"+algo1+"_V_manu_speed.png",(((dej5-np.min(dej5))/(np.max(dej5)-np.min(dej5))*255).astype(np.uint8)))
io.imsave("dej_"+algo1+"_V_manu_composed.png",(((dej6-np.min(dej6))/(np.max(dej6)-np.min(dej6))*255).astype(np.uint8)))

io.imsave("dej_"+algo2+"_Chelsea_const.png",(dej1/256).astype(np.uint8))
io.imsave("dej_"+algo2+"_Chelsea_speed.png",(dej2/256).astype(np.uint8))
io.imsave("dej_"+algo2+"_Chelsea_composed.png",(dej3/256).astype(np.uint8))
io.imsave("dej_"+algo2+"_V_manu_const.png",(((dej4dr-np.min(dej4dr))/(np.max(dej4dr)-np.min(dej4dr))*255).astype(np.uint8)))
io.imsave("dej_"+algo2+"_V_manu_speed.png",(((dej5dr-np.min(dej5dr))/(np.max(dej5dr)-np.min(dej5dr))*255).astype(np.uint8)))
io.imsave("dej_"+algo2+"_V_manu_composed.png",(((dej6dr-np.min(dej6dr))/(np.max(dej6dr)-np.min(dej6dr))*255).astype(np.uint8)))

print("results : "+algo1)
print("   PSNR  ||  SSIM   || absq0.5 || time")
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)-1],dej1,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)-1],dej1,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift1[12:-12]-d1)),
      " || %2.4f"%t1)
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)],dej2,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)],dej2,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift2[12:-12]-d2)),
      " || %2.4f"%t2)
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)],dej3,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)],dej3,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift3[12:-12]-d3)),
      " || %2.4f"%t3)
print("%2.4f"%peak_signal_noise_ratio(im2,dej4,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,dej4,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift4[12:-12]-d4)),
      " || %2.4f"%t4)
print("%2.4f"%peak_signal_noise_ratio(im2,dej5,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,dej5,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift5[12:-12]-d5)),
      " || %2.4f"%t5)
print("%2.4f"%peak_signal_noise_ratio(im2,dej6,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,dej6,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift6[12:-12]-d6)),
      " || %2.4f"%t6)

print("results : "+algo2)
print("   PSNR  ||  SSIM   || absq0.5 || time")
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)-1],dej1dr,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)-1],dej1dr,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift1[12:-12]-d1dr)),
      " || %2.4f"%t1dr)
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)],dej2dr,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)],dej2dr,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift2[12:-12]-d2dr)),
      " || %2.4f"%t2dr)
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)],dej3dr,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)],dej3dr,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift3[12:-12]-d3dr)),
      " || %2.4f"%t3dr)
print("%2.4f"%peak_signal_noise_ratio(im2,dej4dr,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,dej4dr,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift4[12:-12]-d4dr)),
      " || %2.4f"%t4dr)
print("%2.4f"%peak_signal_noise_ratio(im2,dej5dr,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,dej5dr,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift5[12:-12]-d5dr)),
      " || %2.4f"%t5dr)
print("%2.4f"%peak_signal_noise_ratio(im2,dej6dr,data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,dej6dr,data_range = 2**memory_range-1),
      " || %2.4f"%np.median(np.abs(true_shift6[12:-12]-d6dr)),
      " || %2.4f"%t6dr)

comp1 = io.imread("Images/jit_const_Chelsea_ch0_DEJIT.tif").astype(np.float64)
comp2 = io.imread("Images/jit_composed_Chelsea_ch0_DEJIT.tif").astype(np.float64)
comp3 = io.imread("Images/jit_speed_Chelsea_ch0_DEJIT.tif").astype(np.float64)
comp4 = io.imread("Images/jit_const_V_manu_ch0_DEJIT.tif").astype(np.float64)
comp5 = io.imread("Images/jit_composed_V_manu_ch0_DEJIT.tif").astype(np.float64)
comp6 = io.imread("Images/jit_speed_V_manu_ch0_DEJIT.tif").astype(np.float64)

algo3 = "Nam"

print("results : "+algo3)
print("   PSNR  ||  SSIM")
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)-1],comp1[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)-1],comp1[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1))
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)],comp2[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)],comp2[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1))
print("%2.4f"%peak_signal_noise_ratio(im1[:,:-int(true_I)],comp3[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im1[:,:-int(true_I)],comp3[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1))
print("%2.4f"%peak_signal_noise_ratio(im2,comp4[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,comp4[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1))
print("%2.4f"%peak_signal_noise_ratio(im2,comp5[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,comp5[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1))
print("%2.4f"%peak_signal_noise_ratio(im2,comp6[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1),
      " || %2.4f"%structural_similarity(im2,comp6[:,search_zone_length:-search_zone_length],data_range = 2**memory_range-1))

io.imsave("dej_"+algo3+"_Chelsea_const.png",(comp1[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("dej_"+algo3+"_Chelsea_speed.png",(comp2[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("dej_"+algo3+"_Chelsea_composed.png",(comp3[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("dej_"+algo3+"_V_manu_const.png",(comp4[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("dej_"+algo3+"_V_manu_speed.png",(comp5[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))
io.imsave("dej_"+algo3+"_V_manu_composed.png",(comp6[:,search_zone_length:-search_zone_length]/256).astype(np.uint8))

#%% big real image of 15000 lines x 10016 columns
pancreas = io.imread("Images/pancreas.png").astype(np.float64)
search_zone_length = 20
subpixels_number = 10
dej_DP_pancreas,d_DP_pancreas,t_DP_pancreas = DP.main_DP_on_the_grid_L1_regul2(pancreas[:,::-1],search_zone_length,subpixels_number,70000, show = True)
w_WR_pancreas,dej_WR_pancreas,t_WR_pancreas = W_Relax.main_solver_weight_relax_with_pre_computer(pancreas[:,::-1],search_zone_length,1e8, tol = 1e-4, show = True)#reversing the column order as the shift is in the other direction
d_WR_pancreas = W_Relax.shift_from_w(np.reshape(w_WR_pancreas,(-1,search_zone_length)),100)
dej_DP_pancreas = dej_DP_pancreas[:,::-1] #reversing the column order as the shift is in the other direction
dej_WR_pancreas = dej_WR_pancreas[:,::-1] #reversing the column order as the shift is in the other direction
tls.plot(pancreas[2150:2450,5160:5500], "Crop of pancreas image")
tls.plot(dej_DP_pancreas[2150:2450,5160:5500], "Crop of the dejittered pancreas image using the DP algorithm")
tls.plot(dej_WR_pancreas[2150:2450,5160:5500], "Crop of the dejittered pancreas image using the Weight Relaxation algorithm")
tls.plot(np.reshape(w_WR_pancreas, (-1, search_zone_length)).T, "Weights solutions of the Weight Relaxation algorithm")
plt.figure()
plt.plot(-d_DP_pancreas[::-1], label = "DP")
plt.plot(-d_WR_pancreas[::-1], label = "WR")
plt.legend()
plt.title("shifts obtained")
plt.show()
print("time for DP : ",t_DP_pancreas,"time for Weight Relaxation : ",t_WR_pancreas)
io.imsave("demo_pancreas.png",np.clip(pancreas[2150:2450,5160+search_zone_length:5500+search_zone_length],0,2**16-1).astype(np.uint16))
io.imsave("demo_dej_DP_pancreas.png",np.clip(dej_DP_pancreas[2150:2450,5160:5500],0,2**16-1).astype(np.uint16))
io.imsave("demo_dej_WR_pancreas.png",np.clip(dej_WR_pancreas[2150:2450,5160:5500],0,2**16-1).astype(np.uint16))
