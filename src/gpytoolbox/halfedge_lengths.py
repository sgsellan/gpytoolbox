import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared

def halfedge_lengths(V,F):
    # Given a triangle mesh V,F, returns the lengths of all halfedges.
    #
    # The ordering convention for halfedges is the following:
    # [halfedge opposite vertex 0,
    #  halfedge opposite vertex 1,
    #  halfedge opposite vertex 2]
    #
    # Inputs:
    #       V  #V by 3 numpy array of mesh vertex positions
    #       F  #F by 3 int numpy array of face/edge vertex indices into V
    # Outputs:
    #       l  3*#F lengths of halfedges
    
    return np.sqrt(halfedge_lengths_squared(V,F))
