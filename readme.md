This file contains the implementation of the triplet Markov random field presented in the paper:

> Ouali S, Courbot J-B, Pierron R and Haeberle O (2024), "Bayesian image segmentation under varying blur with triplet Markov random field", Inverse Problems. Vol. 40, pp. 095010. IOP Publishing.

The preprint of this paper is available [here](https://hal.science/hal-04660805).

The main entry point of this package is the notebook 'Segmentation example.ipynb', that can be run to test the implemented algorithm. It requires 
having a 3D PSF ('PSF GL.tif') and an image ('90.png') that are both provided.

The package also contains the following files:
- TMRF_functions: it contains the necessary functions for the implementation of the TMRF model 
using the chromatic Gibbs sampler. The file also contains the implementation of the MAP and MPM
estimators
- GMRF_functions: it contains the necessary functions for the simulation of a Gaussian random field
- MH_functions: it contains the necessary functions to implement the Metroplis-Hastings algorithm
- parameter_estimation: this file contations the different function necessary to estimates the
 parameters of the proposed TMRF model 
- Segmentation example: this notebook can be run to test the implemented algorithm. It requires 
having a 3D PSF. An example image named 90.png is also added.
  
