# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:20:37 2024

@author: OUALI

Here you can find the different functions used to simulate a Gaussian random field

"""

# Gaussien field
import numpy as np
from pyfftw.interfaces.numpy_fft import fft2 as pfft2
from pyfftw.interfaces.numpy_fft import ifft2 as pifft2

# =============================================================================
# Segmentation tool for Gaussian random field implementation

# =============================================================================

def gau(x,r) : 
    return np.exp(-(x**2)/r**2)


def get_base_invert_numerical(b):
    '''
    If b is the base of a matrix B, returns bi, the base of B^-1 with the
    direct formula sing Fourier space
    To avoid numerical inf, a mask is used to avoid div by 0s.
    '''
    P, Q = b.shape
    
    bigN = 1.
    NB = pfft2(bigN*b, norm='ortho')

    mask = (NB.real > 1e-6) 
    iNB = np.zeros_like(NB)
    iNB[mask] = np.power(NB[mask],-1)
    if (mask).sum()!=0 :
        iNB[mask==0] = iNB[mask].max()
   
    res = 1 / (P * Q * bigN) * np.real(pifft2(iNB, norm='ortho'))

    return res

def get_base_q(range1,sigma,P,Q,fonc):
    """Compute the precision matrix basis"""
    r = np.zeros((P, Q))
    
    for x in range(P):
        for y in range(Q):
                r[x, y] = sigma**2 * corr_gau(0, x, 0, y, P, Q, range1)
        q = get_base_invert_numerical(r)
    return q

def corr_gau(x1, x2, y1, y2, P, Q, r):
    try:
        return np.exp(-euclidean_dist_torus(x1, x2, y1, y2, P, Q)**2 / r**2)
    except FloatingPointError as e:
        return 0

def euclidean_dist_torus(x1, x2, y1, y2, P, Q):
    '''
    Distance on a torus between two point in the image (left=righht,top=bottom)
    '''
    return np.sqrt(min(np.abs(x1 - x2), P - np.abs(x1 - x2)) ** 2 +
                   min(np.abs(y1 - y2), Q - np.abs(y1 - y2)) ** 2)


def fourier_sampling_gaussian_field(r, P, Q):
    '''
    '''
    X = np.zeros((P, Q), dtype=int)
    Z = np.random.normal(X, 1)
    Z = Z.astype(np.complex128)
    Z += 1j * np.random.normal(X, 1)
    Z = Z.reshape((P, Q))
    #Compute the (real) eigen valuesi matrix in Fourier space
    L = np.fft.fft2(r)
    
    #RUE:
    Y = np.real(np.fft.fft2(np.multiply(np.power(L, -0.5), Z), norm="ortho"))

    return Y

def sample_from_fourier(qb,mb,P,Q):
    """GMRF sampling + mean"""
    
    X = mb+fourier_sampling_gaussian_field(qb, P, Q)
    
    return X