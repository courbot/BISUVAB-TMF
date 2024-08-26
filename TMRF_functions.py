# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:24:36 2024

In the following, you will find all functions 
that are necessary for the implementation of the 
the triplet Markov random field using the chromatic Gibbs
sampler

author: Ouali
"""
import itk
import numpy as np
from scipy.stats import mode
from scipy.fft import rfft2, irfft2
from scipy.fftpack import fftshift

# =============================================================================
# Segmentation tool for TMRF implementation

# =============================================================================

def error_calc(X, X_ref):
    """
    This function that calculates the error between two images
    
    """
    err = ((X_ref != X)*1.0).mean()
    return  min(err,1-err)

def kronecker(a,b):
    """
    The kronecker function:
    It returns one when the value of two pixels is equal
    This function is require for the Potts model implementation

    """
    if (a==b).all():
        return 1
    else:
        return 0
    
       
def get_8_neighbors(img,x,y):
    """
    get_neighbor function returns the 8 neighbors of a pixel. 
    img is the observation image
    x and y are the pixel coordinates (ron and column)

    """

    img_width=img.shape[0]
    img_height=img.shape[1]     
    if x==0:
        if y==0: # Le pixel du coin supérieur à gauche (premier pixel de l'image)
            N=[img[1,0], img[1,1], img[0,1],-1,-1,-1,-1,-1,]
        
        if y==img_height-1:  # Le pixel du coin inférieur à gauche
            N=[img[1,y], img[1,y-1], img[0,y-1],-1,-1,-1,-1,-1]
            
        if y>=1 and y<=img_height-2: #première  colonne 
            N=[img[0,y+1], img[0,y-1], img[1,y], img[1,y-1], img[1,y+1],-1,-1,-1]
         
    if x==img_width-1: 
        if y==0:   # Le pixel du coin supérieur à droite   
            N=[img[x-1,0], img[x-1,1], img[x,1],-1,-1,-1,-1,-1]      
    
        if  y==img_height-1:  # Le pixel du coin inférieur à droite (dernier pixel)        
            N=[img[x-1,y], img[x-1,y-1], img[x,y-1],-1,-1,-1,-1,-1]
            
        if y>=1 and y<=img_height-2:    # dernière  colonne
            N=[img[x,y+1], img[x,y-1], img[x-1,y], img[x-1,y-1], img[x-1,y+1],-1,-1,-1]
    
    
    if x>=1 and x<=img_width-2:
        if y==0: #première ligne
            N=[img[x-1,0],img[x+1,0], img[x,1], img[x-1,1], img[x+1,1],-1,-1,-1]
         
        if y==img_height-1:   # dernière ligne        
            N=[img[x-1,y], img[x+1,y], img[x,y-1], img[x-1,y-1], img[x+1,y-1],-1,-1,-1]
        
        else:
            N=[img[x+1,y], img[x-1,y], img[x,y-1], img[x,y+1], img[x-1,y-1], img[x-1,y+1], img[x+1,y-1], img[x+1,y+1]]
    
    return np.array(N)

# the subdivision function returns a 3D array of the 8 neighbors of the image
def subdivision(img):  
      
    (S0,S1)=img.shape
    voisins = np.zeros((S0,S1,8)) 
    #here we calculate the 8 neighbors for each site and store them in a table
    for p_x in range(S1):
        for p_y in range(S0):
           
            voisins[p_x,p_y,:]=np.array(get_8_neighbors(img,p_x,p_y))
               
    return voisins


# psf reading
def psf_read(filename):
    """
    this function reads the 3D PSF generated using PSFGenerator from imageJ
    """
    psf=itk.imread(filename)
    my_psf = np.array(psf)

    return my_psf

def padding(X, my_psf):
    """
    The padding function is used to adjust the PSF size to the image size
    X is the image
    my_psf is the 3D psf
    """
    
    Q_x,P_x,a_x=X.shape
    a_psf,Q_psf,P_psf=my_psf.shape
    
    if Q_x<Q_psf:
        desired_rows=Q_x
        desired_cols=P_x
        x_pad=int((Q_psf-desired_rows)/2)
        y_pad=int((P_psf-desired_cols)/2)
        
        my_psf_pad=my_psf[:,x_pad:(Q_psf-x_pad), y_pad:(P_psf-y_pad)]
        
    else:
        
        desired_rows=Q_x
        desired_cols=P_x
        my_psf_pad=np.zeros((a_psf,P_x,Q_x))
        for c in range(a_psf):
            my_psf_pad[c,:,:] =np.pad(my_psf[c,:,:], ((int((desired_rows-Q_psf)/2),int((desired_rows-Q_psf)/2)), (int((desired_cols-P_psf)/2), int((desired_cols-P_psf)/2))), 'constant', constant_values=0)
         
    return my_psf_pad


def conv_RGB_f(X,my_psf,V_discret, nbr_channels): 
    """
    The function conv_RGB_f is used to add varying blur to the image
    the calculation are done in the Fourier domain
    """
    #we first pad the psf size to the image size
    my_psf_pad=padding(X, my_psf)
     
    P=X.shape[1]
    Q=X.shape[0]
    a=my_psf.shape[0]
    
    #Res is vector where we stock the final blurred image
    Res=np.zeros_like(X)*1.0
    
    #blur is added to the image channel by channel
    for c in range(nbr_channels):
        #Inter is an intermdiate vector
        Inter=np.zeros((P,Q))
        X_f=rfft2(X[:,:,c])    

        if isinstance(V_discret, int):
            # v constant
            conv_tout=irfft2(rfft2(fftshift(my_psf_pad[V_discret]))*X_f).flatten()
            Inter=(conv_tout.flatten()).reshape(P,Q)
            
        else:
            #v is Gaussian field
            for z in range(a):  
                conv_tout=irfft2(rfft2(fftshift(my_psf_pad[z]))*X_f).flatten()
                mask=V_discret.astype(int)==z
                Inter[mask]=conv_tout[mask.flatten()]
            
            Res[:,:,c]=Inter

    return Res

# =============================================================================
# Gibbs chromatic sampler

# =============================================================================


#Here we calculate the Potts distribution
def dist_grid(k, x_nei_markov_grid, alpha):
    """
    k is the number of classes
    alpha is the granularity
    x_nei_markov_grid is the 8 neighbors of the Markov model
    """
      
    prior = - alpha * (1 - 2 * (k==x_nei_markov_grid).sum(axis=2))
    
    return np.exp(prior)

#Here we calculate the likelihood
def likelihood_grid(class_c, option, Y_grid, H_X_grid, sigma, P_grid, Q_grid):
    """
    option is for the choice of the covariance/variance function
    Y_grid is the observation image
    H_X_grid is the blurred class image
    sigma is wether the variance or covariance
    P_grid and Q_grid are the size of the image grid
    """
    if option==True:#sigmaclasses=true
        likelihood_vect_all=-0.5*((Y_grid-H_X_grid).dot(sigma[class_c])).dot((Y_grid-H_X_grid).T)
        
        likelihood_vect=np.diag(likelihood_vect_all)
        likelihood=np.exp(likelihood_vect.reshape(P_grid,Q_grid))
        
    else:
        likelihood_vect=  -0.5 * ((Y_grid-H_X_grid)**2).sum(axis=1) / sigma[class_c]**2 
        likelihood=np.exp(likelihood_vect.reshape(P_grid, Q_grid))
                    
            
    return likelihood


def gibbs_chromatic(V,Y,mu,sigma,alpha,my_psf,nbrIter, option, X_init = None, ICM=True):
    """
    V is the blur field, Y is the observation, X_init is the initial class field
    
    mu is the mean and sigma the variance/covariance matrix 
    (option=True ==> covariance
     option = False ==> variance)
    
    alpha is the granularty of the Potts distribution
    
    ICM=True we ran the ICM algorithm
    ICM= False we ran the Gibbs algorithm 
    
    my_psf is the 3D psf of the image aquisition system
    # """
    # We will create a random field X0 which will represent the starting random field
    
    (Q,P,a)= Y.shape  
    X_sequence = np.zeros(shape=(P,Q,nbrIter)) # table where estimated fields are stocked
    nbr_channels=a # whether it is a gray scale image or an RGB image

    indices=np.arange(len(mu))
    # np.random.seed(0)
    if X_init is not None :
        X = X_init.copy()
    else:
        X=np.array(np.random.choice(indices, size=(P,Q)))
        
    nbr_grid=4 #this represent the number of colors (sites) for the chromatic sampler
    pas_s=2 # the seperation between pixels of the same site (color)
    
    li_grid=np.array(([[0,0,1,1], [0,1,0,1]])) 
    
    #Y does not change so we can put it in grid form before starting the iterations
    Y_grid = np.zeros(shape=(int(P/pas_s),int(Q/pas_s),nbr_channels,nbr_grid))
    for q in range(nbr_grid):
        Y_grid[:,:,:,q] = Y[li_grid[0,q]::pas_s,li_grid[1,q]::pas_s]
        
        
    #iterat is the number of iterations    
    for iterat in range(nbrIter): 
        #s is the grid on which we work                          
        for s in range(nbr_grid):
            neighbors =subdivision(X) #here we get the neighbors of X
            neighbors_grid=neighbors[li_grid[0,s]::pas_s, li_grid[1,s]::pas_s]# here we put the neighbors into grids 
            P_grid=neighbors_grid.shape[1]
            Q_grid=neighbors_grid.shape[0]
           
            prior= np.zeros(shape=(P_grid,Q_grid,len(mu)))#table to stock the prior 
            likelihood=np.zeros_like(prior)#table to ctock the likelihood
            X_colored=mu[X]#colored version of the class field
            
            for c in range(len(mu)):
                #here we calculate the prior
                
                prior[:,:,c]= dist_grid(c, neighbors_grid, alpha)
                
                #here we calculate the likelihood
                X_colored[li_grid[0,s]::pas_s, li_grid[1,s]::pas_s]=mu[c]
                H_X=conv_RGB_f(X_colored,my_psf,V,nbr_channels)
                H_X_grid=H_X[li_grid[0,s]::pas_s, li_grid[1,s]::pas_s]
                
                
                Y_grid_vect=Y_grid[:,:,:,s].reshape((P_grid*Q_grid,nbr_channels))
                H_X_grid_vect=H_X_grid.reshape((P_grid*Q_grid,nbr_channels))
                
                likelihood[:,:,c]=likelihood_grid(c, option, Y_grid_vect, H_X_grid_vect, sigma, P_grid, Q_grid)
            
            likelihood/=likelihood.sum(axis=2)[:,:,np.newaxis]#normalization  
            prior/=prior.sum(axis=2)[:,:,np.newaxis]#normalization
            probas=(prior*likelihood)
            probas /=probas.sum(axis=2)[:,:,np.newaxis]#normalization
                       
            r =  np.random.random(size=(probas.shape[0],probas.shape[1]))
            cum_sum = np.cumsum(probas, axis=2)              
#           recreating the original image
            if ICM==True:
                X[li_grid[0,s]::pas_s, li_grid[1,s]::pas_s]=np.argmax(probas, axis=2)
            else:
                X[li_grid[0,s]::pas_s, li_grid[1,s]::pas_s]=(r[:, :, np.newaxis] < cum_sum).argmax(axis=2)
                

        X_sequence[:,:,iterat] = X
        if iterat > 10:
            avg_last_10 = X_sequence[:,:,iterat-10:iterat].mean(axis=2)#avg
            #avg_last_10,_ = mode(X_sequence[:,:,iter-10:iterat],axis=2)#mode
            diff = error_calc(X,avg_last_10) # avg or mode
            
            if diff < 0.05:
                print("gibbs chromatic stops at %.0f"%iterat)
                break; # Gibbs is done
            
    return mu[X], X


# =============================================================================
# Segmentation tool
# MPM Criterio 
# =============================================================================

def MPM(V,Y,X_prec,mu,sigma,alpha,my_psf,nbr_Iter_gibbs,nbr_Iter_mpm, option):
    
    (Q,P,a)=Y.shape
    X_stock = np.zeros(shape=(P,Q,nbr_Iter_mpm))

    for m in range(nbr_Iter_mpm): 
        print("Iteration %.0f"%m+1)
        _, X_stock[:,:,m] = gibbs_chromatic(V,Y ,mu,sigma,alpha,my_psf,nbr_Iter_gibbs, option, X_init = X_prec, ICM=False)


    X_mpm = X_stock.mean(axis=2)>0.5 
    
    X_RGB=mu[X_mpm*1]
    
    return X_mpm, X_RGB


# =============================================================================
# Segmentation tool
# MAP Criterio 
# =============================================================================

def MAP_segmentation(Y,X_prec, V, my_psf,sigma, mu, alpha,nbr_iter_gibbs, option):
    
    X_RGB,X_map=gibbs_chromatic(V,Y,mu,sigma,alpha,my_psf,nbr_iter_gibbs,option,X_init =X_prec, ICM=True)     

    return X_map, X_RGB
