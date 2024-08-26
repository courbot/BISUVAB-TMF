# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:14:35 2024

@author: e1803267
The different functions envolved in the implementation of the MH algorithm 
can be found below
"""

import numpy as np
from pyfftw.interfaces.numpy_fft import fft2 as pfft2
from pyfftw.interfaces.numpy_fft import ifft2 as pifft2
import TMRF_functions as tmrf
import GMRF_functions as gmrf
import scipy.linalg as sl
# =============================================================================
# Segmentation tool for Metropolis-Hastings (MH) implementation

# =============================================================================


def scaler(V, minimum, maximum):
    """
    this function returns a normalized version of the Gaussian field V 
    the minimum value is 0
    the maximum value is the PSF size in the third dimension (number of sections)
    """
    return np.round(((V-V.min())/(V.max()-V.min()))*(maximum - minimum) + minimum)


##
def log_prob(X_colored, X, Y, V, my_psf, q, m, sigma, mu, option):
    """
    this function returns the distribution 
    X_colored is the colored version of the class field X
    Y is the observation
    V is the Gaussian field representing the blur
    my_psf is the PSF
    q is the precision matrix
    m is the mean of the Gaussian field (it is equal to zero)
    sigma is the variance 
    mu represent the colors (classes) of the observation Y
    
    """
    #constants
    
    a=my_psf.shape[0]
    nbr_channels=Y.shape[2]
    ly,lx=q.shape
    
    #Calculation in the Fourier domain of the prior distribution
    
    Qf=pfft2(q)
    Vf=pfft2(V)
    QV = pifft2(Qf * Vf)
    
    prior = (-0.5 * (V * QV).sum()).real
    
    
    #Calculation of the likelihood
    V=scaler(V, 0, a-1) # normalization
    HX=tmrf.conv_RGB_f(X_colored, my_psf, V, nbr_channels)
    
    if option==False:
        
        likelihood=-(1/sigma**6) * ((np.linalg.norm(Y-HX))**2)
        
    else:
    
        X2=np.reshape(X,(lx*ly))
        
        HX2=np.reshape(HX,(lx*ly,3))
        Y2=np.reshape(Y,(lx*ly,3))
        
        likelihood=[]
        indice=np.arange(len(mu))
        for i in indice:
            i2=np.tile(i,(lx*ly))   
            likelihoodAB= np.linalg.norm(np.dot((Y2[X2==i2]-HX2[X2==i2]), sl.sqrtm(np.linalg.inv(sigma[i]))))**2
            likelihood.append(likelihoodAB)
        
    return prior+np.sum(likelihood)

def proposal(v, q, m, stepsize):
    """
    This function return a proposal of the Gaussian field V
    V_proposal = v_actual + v_perturbation*stepsize
    """
    ly,lx=q.shape
    return v+gmrf.sample_from_fourier(q,m,lx,ly)*stepsize

def p_acc_MH(v_new, v_old, X, X_BW, Y, my_psf, q, m, sigma,mu, option):
    """
    calculation of the probability of acceptance
    """
    return min(1, np.exp(log_prob(X,X_BW,Y,v_new,my_psf,q,m,sigma,mu, option) - log_prob(X,X_BW,Y,v_old,my_psf,q,m,sigma,mu, option)))

def sample_MH(v_old, stepsize, X,X_BW, Y, H, q, m, sigma,mu, lp_old, option):# ajout lp_old
    """
     here we determine whether we accept the new state or not:
     we draw a random number uniformly from [0,1] and compare
     it with the acceptance probability
    """

    v_new = proposal(v_old, q, m, stepsize)
    
    lp_new=log_prob(X, X_BW, Y,v_new,H,q,m,sigma, mu, option)
    
    if lp_new > lp_old:   
        accept = True
        
    else:  
        accept = np.random.random() < np.exp(lp_new - lp_old) 
    
    if accept:
        return accept, v_new, lp_new
    else:
        return accept, v_old, lp_old

def build_MH_chain(init, stepsize, n_total, X_colored, X, Y, my_psf, q, m, sigma, mu, option):
    '''
    implementation of the MH algorithm
    n_total is the number of iterations
    stepsize is the disturbance amplitude
    init is the initail Gaussian field
    
    
    '''
    n_accepted = 0
    chain = [init]
    log_proba=[log_prob(X_colored, X, Y,init,my_psf,q,m,sigma, mu, option)]
    for _ in range(n_total):
        accept, state, proba = sample_MH(chain[-1], stepsize, X_colored,X, Y, my_psf, q, m, sigma, mu, log_proba[-1], option)
        chain.append(state)
        n_accepted += accept
        log_proba.append(proba)

    acceptance_rate = n_accepted / float(n_total)

    return chain, acceptance_rate,log_proba