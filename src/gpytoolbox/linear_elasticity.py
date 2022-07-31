import numpy as np
from scipy.sparse import csr_matrix

import gpytoolbox
from .linear_elasticity_stiffness import linear_elasticity_stiffness
import sys
import os
from .min_quad_with_fixed import min_quad_with_fixed


def linear_elasticity(V,F,U0,dt=0.1,bb=None,bc = None
    ,Ud0=None,fext=None,K=1.75,mu=0.0115,volumes=None,mass=None):
    # Compute the deformation of a 2D solid object according to the usual linear elasticity model 
    #
    # Note: This only works for 2D (d=2) meshes currently
    # TO-DO: Code tet mesh version of this
    #
    # Inputs:
    #       V #V by d numpy array of vertex positions 
    #       F #F by d+1 integer numpy array of element indeces into V
    #       U0 #V by d numpy array of previous displacement
    #       Optional:
    #           dt float timestep
    #           bb #bb integer numpy array of fixed vertex indeces into V
    #           bc #bb by d numpy array of fixed vertex coordinates
    #           K bulk modulus
    #           mu material shear modulus
    #           volumes an #F numpy array with the volumes (if d=3) or areas (if d=2) of each mesh element
    #           mass the shape's #V by #V scipy sparse mass matrix (will be computed otherwise)
    #           fext #V by d external forces (for example, gravity or a load)
    #           Ud0 #V by d numpy array of previous velocity
    # Outputs:
    #       U  #V by d numpy array of new displacements
    #       sigma_v #F numpy array of Von Mises stresses

    if (Ud0 is None):
        Ud0 = 0*V
    if (fext is None):
        fext = 0*V
    if (bb is not None):
        # THIS ASSUMES 2D
        bb = np.concatenate((bb,bb+V.shape[0]))
        bc = np.concatenate((bc[:,0],bc[:,1]))
    
    K, C, strain, A, M = linear_elasticity_stiffness(V,F,K=K,volumes=volumes,mass=mass,mu=mu)

    
    A = M + (dt**2)*K
    B = M*((dt**2)*np.reshape(fext,(-1, 1),order='F') + np.reshape(U0,(-1, 1),order='F') + dt*np.reshape(Ud0,(-1, 1),order='F'))

    U = min_quad_with_fixed(A,c=-1.0*np.squeeze(B),k=bb,y=bc)
    #print(U)
    # https://en.m.wikipedia.org/wiki/Von_Mises_yield_criterion
    face_stress_vec = C*strain*U
    if V.shape[1]==2:
        sigma_11 = face_stress_vec[0:F.shape[0]]
        sigma_22 = face_stress_vec[F.shape[0]:(2*F.shape[0])]
        sigma_12 = face_stress_vec[(2*F.shape[0]):(3*F.shape[0])]
        sigma_v = np.sqrt(0.5*(sigma_11*sigma_11 - sigma_11*sigma_22 + sigma_22*sigma_22 + 3*sigma_12*sigma_12))
    elif V.shape[1]==3:
        print("TET MESHES UNSUPPORTED")
    return U, sigma_v