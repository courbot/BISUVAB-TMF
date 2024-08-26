# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:39:57 2024
@author: OUALI

In the following, you will find the functions used for 
the parameter estimation.

These functions can be used in EM or SEM algorithm or 
in SPE algorithm.

"""

import TMRF_functions as tmrf
import GMRF_functions as gmrf
import MH_functions as mh
from scipy.stats import mode
from scipy.optimize import curve_fit
import scipy.signal as si
import numpy as np


def sigma_estimate(X_colored,Y, my_psf, V):
    """
    This function returns the variance estimate using MLE
    
    X is the class image
    Y is the observation image
    V is the blur field
    my_psf is the 3D PSF of the aquisition system
    
    """
    (h,l,a)=Y.shape
    (h2,l2,a2)=X_colored.shape
    
    HX=tmrf.conv_RGB_f(X_colored, my_psf, V, a)
    
    
    #we change the sizes of X and Y 
    Y2=np.reshape(Y,(h*l,a))
    HX2=np.reshape(HX,(h2*l2,a2))
       
    std=np.std(Y2.T-HX2.T)
    return std

def sigma_estimate_V(V):
    """
    This function returns the variance estimate of the blur field V using MLE
    
    V is the blur field
    
    """

    return np.std(V)

# def covariance_estimate(X, X_colored, Y, my_psf, V, K):
#     """
#     This function returns the covariance estimate using MLE
    
#     X is the class image
#     X_colored is a colored version of X
#     Y is the observation image
#     V is the blur field
#     my_psf is the 3D PSF of the aquisition system
#     K is the number of classes
    
#     """
    
#     (h2,l2,a2)=X.shape
#     (h,l,a)=Y.shape
    
    
#     HX=tmrf.conv_RGB_f(X, my_psf, V,a)
    
#     #we change the sizes of X and Y 
#     Z=np.reshape(Y,(h*l,a))
#     P=np.reshape(HX,(h2*l2,a2))
#     U=np.reshape(X,(h2*l2))
    
#     covariance=[]
#     indice=np.arange(K)
    
#     for i in indice:
#         cov=np.cov(Z[U==i].T-P[U==i].T)
#         covariance.append(cov)
#     return np.asarray(covariance)



def covariance_estimate(X, X_colored, Y, my_psf, V, K):
        """
        This function returns the covariance estimate using MLE
       
        X is the class image
        X_colored is a colored version of X
        Y is the observation image
        V is the blur field
        my_psf is the 3D PSF of the aquisition system
        K is the number of classes
       
        """
        (h,l,a)=Y.shape
        (h2,l2,a2)=X_colored.shape

        HX=tmrf.conv_RGB_f(X_colored, my_psf, V,a)
        
        #we change the sizes 
        Y2=np.reshape(Y,(h*l,a))
        X2=np.reshape(X,(h2*l2))
        HX2=np.reshape(HX,(h2*l2,a2))
       
        stds=np.zeros((K,a))
        for class_i in range(K):
            for canal in range(a):    
                stds[class_i, canal]=np.std(Y2[X2==class_i].T-HX2[X2==class_i].T)
        
        covariance=[]
        for i in range(K):       
            covariance.append(np.diag(stds[i,:]))
        return np.asarray(covariance)


def mean_estimate(X_colored,Y,K):
    """
    This function returns the mean estimate using MLE
    
    X is the class image
    Y is the observation image
    K is the number of classes
    """
    (P,Q,a)=Y.shape
    
    Y=Y.reshape(P*Q,a)
    
    mean_new=np.zeros((K,a))
    for class_i in range(K):
        mean_new[class_i]=Y[(X_colored==class_i).flatten(),:].mean(axis=0)
    return mean_new


def MLE(X, X_colored, Y, K, P, Q, my_psf, V, option):

    if option== False:
        sigma=sigma_estimate(X_colored.copy(),Y,my_psf, V)
    else:
        sigma=covariance_estimate(X.copy(),X_colored, Y, my_psf, V, K)
 
    mu=mean_estimate(X, Y, K)
 
    alpha=alpha_estimate(X,K,P,Q)
    
    return sigma, mu, alpha


def alpha_estimate(X,K,P,Q): 
    
    """
    This function returns the mean estimate using MLE
    
    X is the class image
    K is the number of classes
    P,Q is the size of X
    """
    
    t=0
    
    for p_y in range(Q): 
        for p_x in range(P-1):
            
            t=t+tmrf.kronecker(X[p_x][p_y],X[p_x +1][p_y])  
            
    freq=t/(P*Q)
    
    alpha=(K**2/(2*(K-1)))*(freq-(1/K))
    return alpha


def est_r(V):
    """
    This function estimate the correlation range, rho, from a GMRF realization
    using LS estimator
    Source : Gangloff et al 21
    """
    
    P,Q = V.shape
    mv = V.mean()
    autocorr = si.fftconvolve(V-mv,V[::-1,::-1]-mv,mode = 'same')[::-1,::-1]
    
    autocorr /= autocorr.max()

    C = np.zeros(shape=(P))
    nb_pt = np.zeros_like(C)
    
    dx,dy = np.ogrid[:P,:Q]

    D = (np.sqrt((dx-P/2+1)**2 + (dy-Q/2+1)**2))
    D = np.round(D).astype(int)
    
    for a in range(autocorr.shape[0]):
        for b in range(autocorr.shape[1]): 
            
            index = D[a,b]
            C[index] += autocorr[a,b] 
            nb_pt[index] +=1

    lim = int(P/2)
    correlogram = C[:lim]/nb_pt[:lim]

    where_zero = np.where(correlogram<=0)[0]
    if len(where_zero)==0: l = int(P/2)
    else: l=where_zero[0]
    
    dxc = np.arange(correlogram.size)
    r_est, _ = curve_fit(gmrf.gau, dxc[:l], correlogram[:l])
    
    return max(r_est[0],2)  

# =============================================================================
# Segmentation tool
# SPE Algorithm 
# =============================================================================

def SPE(Y,X_prec,V, my_psf,sigma, mu, alpha,q,radius,step_size,n_total,iter_SPE, option):
    a, P, Q = my_psf.shape
    K=len(mu)
    V_seq=[V]
    X_seq=[]
    
    for i in range(iter_SPE):
        print("iteration numéro", i+1)
        X_colored,X_BW=tmrf.gibbs_chromatic(V,Y,mu,sigma,alpha, my_psf, i+1, option, X_init = X_prec, ICM=False)
        X_seq.append(X_BW)
        
        V_est,_,_ = mh.build_MH_chain(V, step_size, n_total, X_colored, X_BW, Y, my_psf, q, 0, sigma, mu, option)
        V=mh.scaler(V_est[-1],0,a-1)
        V_seq.append(V.copy())      
        
        sigma, mu, alpha=MLE(X_BW, X_colored, Y, K, P, Q, my_psf, V, option)
        rho = est_r(V)
        sigma_V=sigma_estimate_V(V)
        q = gmrf.get_base_q(rho,sigma_V,P,Q,"gau")
        
        if i > 10:
            avg_last_10,_ = mode(X_seq[i-10:i])
            taux_variation_X = tmrf.erreur(X_BW,avg_last_10)   
            if taux_variation_X < 0.05:#                        
                print("SPE stops at %.0f"%i)
                break;
    return sigma, mu, alpha, V_est[-1]