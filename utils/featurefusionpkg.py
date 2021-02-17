#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:22:45 2019

@author: root
"""
import math
import numpy as np

# Scipy Libraries
import scipy.ndimage

# Scikit-image modules
from skimage.transform import resize
from skimage.restoration import denoise_tv_bregman

# Scikit-learn modules
from sklearn.metrics import confusion_matrix


def normalize_data(InData):
    InData.astype(float)
    a       = np.amin(InData)
    b       = np.amax(InData)
    scale   = 1.0 / (b - a)
    OutData = np.multiply(scale, np.subtract(InData,a))
    return OutData



def gaussian_kernel(window_size,blur_sigma):
    t      = np.linspace(-(window_size//2),(window_size//2),window_size)
    x, y   = np.meshgrid(t,t)
    kernel = 1/(2*np.pi*(blur_sigma**2)) * np.exp(-(np.power(x,2) + np.power(y,2))/(2*(blur_sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel




def spatial_blurring(Io,p,window_size,blur_sigma,dec_type):

    M, N, L = Io.shape
    M_hs    = M // p
    N_hs    = N // p
    L_hs    = L    
    I_hs    = np.zeros([M_hs,N_hs,L_hs])
    
    if (dec_type == 'gaussian_filter'):
        kernel = gaussian_kernel(window_size,blur_sigma)
        for i in range(0, L):
#            I_temp = gaussian_filter(Io[:,:,i],blur_sigma)
            I_temp = scipy.ndimage.convolve(Io[:,:,i],kernel,mode='nearest')
            I_temp = I_temp[np.arange(0,M,p)]
            I_temp = I_temp[:,np.arange(0,N,p)]
            I_hs[:,:,i] = I_temp;
    if (dec_type == 'sum'):
        for i in range(0, M_hs):
            for j in range(0, N_hs):
                block_temp = Io[i*p:(i+1)*p,j*p:(j+1)*p:]
                block_temp = block_temp.sum(axis=0)
                block_temp = block_temp.sum(axis=0)
                I_hs[i,j,:] = block_temp
    if (dec_type == 'average'):
        for i in range(0, M_hs):
            for j in range(0, N_hs):
                block_temp = Io[i*p:(i+1)*p,j*p:(j+1)*p:]
                block_temp = block_temp.mean(axis=0)
                block_temp = block_temp.mean(axis=0)
                I_hs[i,j,:] = block_temp
    return I_hs




def spectral_blurring(Io,q,dec_type):
    M, N, L = Io.shape
    M_ms    = M
    N_ms    = N
    L_ms    = L // q
    
    I_ms    = np.zeros([M_ms,N_ms,L_ms])    
    
    if (dec_type == 'sum'):
        for i in range(0, L_ms):
            I_temp = Io[:,:,i*q:(i+1)*q]
            I_temp = I_temp.sum(axis=2)
            I_ms[:,:,i] = I_temp        
    if (dec_type == 'average'):
        for i in range(0, L_ms):
            I_temp = Io[:,:,i*q:(i+1)*q]
            I_temp = I_temp.mean(axis=2)
            I_ms[:,:,i] = I_temp   
    return I_ms




def colored_coded_aperture(shots, M, N, L, num_filters):
    band_wide    = L // num_filters
    aux_var      = np.mod(L,num_filters)
    filter_set   = np.zeros([L,num_filters])
        
    #Building filters
    for i in range(0,num_filters):
        if ((i+1) <= num_filters - aux_var):
            d1  = i * band_wide
            d2  = (i+1) * band_wide
            filter_set[d1:d2,i] = 1
        else:
            d1  = i * band_wide + (i - (num_filters - aux_var))
            d2  = d1 + band_wide + 1
            filter_set[d1:d2,i] = 1
                
    # Optimal coding pattern
    filter_disp = np.zeros([M,N,num_filters],dtype=int)
    for i in range(0,M):
        for j in range(0,N):
            filter_disp[i,j,:] = np.random.permutation(num_filters)
                
        
#    print(filter_disp.sum(axis=2))

    # Building colored coded apertures
    cca = np.zeros((shots,), dtype=np.object)
        
    for i in range(0,shots):
        cca[i] = np.zeros([M,N,L])
        for j in range(0,M):
            for k in range(0,N):
                cca[i][j,k,:] = filter_set[:,filter_disp[j,k,i]]
            
    return cca, filter_set, filter_disp




def patterned_shots(I, comp_ratio):
    M, N, L       = I.shape
    num_filters   = math.floor(L * comp_ratio)
    shots         = math.floor(L * comp_ratio)
    
    colored_ca, filter_set, filter_disp = colored_coded_aperture(shots, M, N, L, num_filters)
    
    shot_data     = np.zeros([M,N,shots])
    for i in range(0, shots):
        I_temp           = np.multiply(I,colored_ca[i])
        shot_data[:,:,i] = I_temp.sum(axis=2)
        
    return shot_data, filter_set, filter_disp, shots



def feature_extraction_patterned(shot_data,M,N,shots,filter_disp):
    
    features = np.zeros([M,N,shots])
    for i in range(0,M):
        for j in range(0,N):
            for k in range(0,shots):
                features[i,j,filter_disp[i,j,k]] = shot_data[i,j,k]
        
    features = features.reshape(M*N,shots)
    return features        




def interpolate_hs_features(features,M,N,p):
    num_filters = len(features[1])    
    interpolated_features = np.zeros([M*N*p*p,num_filters])
    
    for i in range(0,num_filters):
        temp = features[:,i]
        temp = temp.reshape(M,N)
        temp = resize(temp,(M*p,N*p),anti_aliasing=False)
        interpolated_features[:,i] = temp.reshape(M*N*p*p)    
    return interpolated_features



def class_indices(ground_truth, training_rate):
    M, N         = ground_truth.shape
    ground_truth = ground_truth.reshape(M*N)
    num_classes  = ground_truth.max(axis=0)
    num_train_samples = np.zeros([num_classes,1],dtype=int)
    num_test_samples  = np.zeros([num_classes,1],dtype=int)
    
    for i in range(1,num_classes + 1):
        class_indices        = np.where(ground_truth == i)
        class_indices        = np.array(class_indices)
        long                 = len(class_indices[0])
        random_permutation   = np.random.permutation(long)
        train_samples        = int(round(long*training_rate))
        test_samples         = long - train_samples
        num_train_samples[i-1] = int(train_samples)
        num_test_samples[i-1]  = int(test_samples)

        if (i==1):
            training_indices = class_indices[0,random_permutation[0:train_samples]]
            test_indices     = class_indices[0,random_permutation[train_samples:long]]
        else:
            training_indices = np.hstack((training_indices, class_indices[0,random_permutation[0:train_samples]]))
            test_indices     = np.hstack((test_indices, class_indices[0,random_permutation[train_samples:long]]))
    
    return training_indices, test_indices, num_train_samples, num_test_samples




def compute_accuracy(Y_test, Y_pred,verbose=False):
    confusion_mat =confusion_matrix(Y_test,Y_pred)
    M, N = confusion_mat.shape
    
    OA             = np.trace(confusion_mat) / np.sum(confusion_mat.reshape(M*N))
    class_acc      = np.divide(np.diag(confusion_mat), np.sum(confusion_mat,axis=1))
    for i in range(0,M):
        if (np.isnan(class_acc[i])):
            np.empty(class_acc[i])
    
    ca_length      = class_acc.shape
    AA             = np.sum(class_acc) / ca_length[0]
    sqrt_den       = np.sum(confusion_mat.reshape(M*N))
    Pe             = np.dot(np.sum(confusion_mat,axis=0), np.sum(confusion_mat,axis=1)) / (sqrt_den*sqrt_den)
    kappa          = (OA-Pe)/(1-Pe)

    if (verbose):
        print("")
        print("Class accuracy:")
        for i in range(0,ca_length[0]):
            print("                  %.4f" %(class_acc[i]))
    
        print("")
        print("Overall accuracy: %.4f" %(OA))
        print("Average accuracy: %.4f" %(AA))
        print("kappa:            %.4f" %(kappa))
    return OA, AA, kappa, class_acc



def otvca(Y,M,N,num_features,lmbd,itmax):
    b           = np.max(Y)
    a           = np.min(Y)
    Y           = (Y - a) / (b - a)
    F_new      = np.zeros([M*N,num_features])
    _, _, V    = np.linalg.svd(Y, full_matrices =  False)
    V          = V.transpose()
    V_old      = V[:,0:num_features]
    
    for k in range(0,itmax):
        G = np.matmul(Y, V_old)
        for i in range(0,num_features):
            F_tmp = G[:,i]
            F_tmp = denoise_tv_bregman(F_tmp.reshape(M,N),1/lmbd)
            F_new[:,i] = F_tmp.reshape(M*N)
            
        P, _, QT = np.linalg.svd(np.matmul(F_new.transpose(), Y), full_matrices =  False)
        V_new = np.matmul(P, QT)
        V_old = V_new.transpose()
    return F_new


def mfca(Y,M,N,num_features,itmax,window_size):
    b           = np.max(Y)
    a           = np.min(Y)
    Y           = (Y - a) / (b - a)
    F_new       = np.zeros([M*N,num_features])
    _, _, V     = np.linalg.svd(Y, full_matrices =  False)
    V           = V.transpose()
    V_old       = V[:,0:num_features]
   
    for k in range(0,itmax):
        G = np.matmul(Y, V_old)
        for i in range(0,num_features):
            F_tmp = G[:,i]
            F_tmp = scipy.ndimage.median_filter(F_tmp.reshape(M,N),window_size)
            F_new[:,i] = F_tmp.reshape(M*N)
            
        P, _, QT = np.linalg.svd(np.matmul(F_new.transpose(), Y), full_matrices =  False)
        V_new = np.matmul(P, QT)
        V_old = V_new.transpose()
    return F_new



def label2color_py(I,dataset):
    M, N = I.shape
    rgb_image = np.zeros([M,N,3],dtype=int)
    
    if (dataset == 'paviaU'):
        colors = np.array([[0, 0, 0],[192, 192, 192],[0, 255, 0],[0, 255, 255],[0, 128, 0],
                          [255, 0, 255],[165, 82, 41],[128, 0, 128],[255, 0, 0],[255, 255, 0]])
    if (dataset == 'pavia'):
        colors = np.array([[0, 0, 0],[0, 0, 255],[0, 128, 0],[0, 255, 0],[255, 0, 0],
                          [142, 71, 2],[192, 192, 192],[0, 255, 255],[246, 110, 0],[255, 255, 0]])    
    if (dataset == 'indian'):
        colors = np.array([[0, 0, 0],[140, 67, 46],[0, 0, 255],[255, 100, 0],[0, 255, 123],
                          [164, 75, 155],[101, 174, 255],[118, 254, 172],[60, 91, 112],[255, 255, 0],
                          [255, 255, 125],[255, 0, 255],[100, 0, 255],[0, 172, 254],[0, 255, 0],
                          [171, 175, 80],[101, 193, 60]])
        
    for i in range(0,M):
        for j in range(0,N):
            rgb_image[i,j,:] = colors[I[i,j]]
                
    return rgb_image

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def hs_matrix_3dcassi(cca_hs, p):
    shots_hs, M_hs, N_hs, L = cca_hs.shape
    for i in range(0,shots_hs):
        cband = np.zeros([M_hs,N_hs])
        for j in range(0,L):
            cband = cca_hs[i,:,:,j]
            cvctr = cband.reshape(M_hs*N_hs,order='F')
            ax    = np.where(cvctr==1)
            ax    = ax[0]
            ay    = np.matlib.repmat(np.transpose(ax),p**2,1)
            a     = np.reshape(np.transpose(ay),(1,ay.shape[0]*ay.shape[1]))
            a     = np.add(a, i*M_hs*N_hs)
            aux1  = np.matlib.repmat(np.arange(p).reshape(p,1),1,ax.shape[0])
            aux2  = np.matlib.repmat(aux1,p,1)
            aux3  = p * np.remainder(ay,M_hs)
            aux4  = np.add(aux2,aux3)
            aux5  = np.zeros(aux4.shape)
            for k in range(0,p):
                aux5[k*p:(k+1)*p,:] = np.add(aux4[k*p:(k+1)*p,:], k*M_hs*p) 
            aux6 = np.add(aux5, (p**2)*M_hs*np.floor_divide(ay,M_hs))
            bz = np.add(aux6, (p**2)*M_hs*N_hs*j)
            b    = np.reshape(bz,(1,bz.shape[0]*bz.shape[1]),order='F')
            if ((j == 0)&(i==0)):
                at = a
                bt = b
            else:
                at = np.hstack((at, a))
                bt = np.hstack((bt, b))
    at = at[0]
    bt = bt[0]
    return at, bt

def ms_matrix_3dcassi(cca_ms,q):
    shots_ms, M, N, L_ms = cca_ms.shape
    for i in range(0,shots_ms):
        cband = np.zeros([M,N])
        for j in range(0,L_ms):
            cband = cca_ms[i,:,:,j]
            cvctr = cband.reshape(M*N,order='F')
            ax    = np.where(cvctr==1)
            ax    = ax[0]
            ay    = np.matlib.repmat(np.transpose(ax),q,1)
            a     = np.reshape(np.transpose(ay),(1,ay.shape[0]*ay.shape[1]))
            a     = np.add(a, i*M*N)
            aux1  = np.zeros(ay.shape)
            for k in range(0,q):
                aux1[k,:] = np.add(ay[k,:], k*M*N)
            bz   = np.add(aux1, j*M*N*q)
            b    = np.reshape(bz,(1,bz.shape[0]*bz.shape[1]),order='F')
            if ((j == 0)&(i==0)):
                at = a
                bt = b
            else:
                at = np.hstack((at, a))
                bt = np.hstack((bt, b))
    at = at[0]
    bt = bt[0] 
    return at, bt



def ms_matrix_cassi(cca_ms,q):
    shots_ms, M, N, L_ms = cca_ms.shape
    for i in range(0,shots_ms):
        cband = cca_ms[i,:,:,0]
        cvctr = cband.reshape(M*N,order='F')
        ax1   = np.where(cvctr==1) 
        ax    = np.transpose(ax1[0]) + i*M*N
        for j in range(0,L_ms*q):
            bx = np.transpose(ax1[0]) + j*M*N
            if ((j == 0)&(i==0)):
                at = ax
                bt = bx
            else:
                at = np.hstack((at, ax))
                bt = np.hstack((bt, bx))
    return at, bt