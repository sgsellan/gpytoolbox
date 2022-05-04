import numpy as np
from scipy.sparse import vstack
from .fd_partial_derivative_octree import fd_partial_derivative_octree

def fd_grad_octree(V,Q):
    # TODO COMMENT THIS
    Dx,staggered_x =  fd_partial_derivative_octree(V,Q,direction=0)
    Dy,staggered_y =  fd_partial_derivative_octree(V,Q,direction=1)
    Dz,staggered_z =  fd_partial_derivative_octree(V,Q,direction=2)
    return vstack((Dx,Dy,Dz)), vstack((staggered_x,staggered_y,staggered_z))