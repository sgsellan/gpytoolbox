import numpy as np
# Bindings using Eigen and libigl:
import sys
import os
from .remove_duplicate_vertices import remove_duplicate_vertices


def bad_quad_mesh_from_quadtree(C,W,CH):
    # BAD_QUAD_MESH_FROM_QUADTREE
    # From a proper quadtree, builds a connected but degenerate quad mesh
    # containing only the leaf nodes, mostly for visualization purposes.
    #
    # V,Q = bad_quad_mesh_from_quadtree(C,W,CH)
    #
    # Inputs:
    #   C #nodes by 3 matrix of cell centers
    #   W #nodes vector of cell widths (**not** half widths)
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #
    # Outputs:
    #   V #V by 3 matrix of vertex positions
    #   Q #Q by 4 matrix of quad indeces into V
    #   H is None if dimension is 2, contains hex indeces if dimension is 3
    #
    # Example:
    #
    #
    # See also: initialize_quadtree
    dim = C.shape[1]
    is_child = (CH[:,1]==-1)
    W = W[is_child][:,None]
    C = C[is_child,:]
    H = None
    # translate this
    Q = np.linspace(0,W.shape[0]-1,W.shape[0],dtype=int)[:,None]
    if dim==2:
        V = np.vstack((
            C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[-1,-1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[-1,1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[1,1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[1,-1]]),(W.shape[0],1)),
            ))
        Q = np.hstack((
            Q,
            Q + W.shape[0],
            Q + 2*W.shape[0],
            Q + 3*W.shape[0]
        ))
    else:
        V = np.vstack((
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[-1,-1,-1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[-1,1,-1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[1,1,-1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[1,-1,-1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[-1,-1,1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[-1,1,1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[1,1,1]]),(W.shape[0],1)),
            C + 0.5*np.tile(W,(1,3))*np.tile(np.array([[1,-1,1]]),(W.shape[0],1))
            ))
        H = np.hstack((Q + 0*W.shape[0],Q + 1*W.shape[0], Q + 2*W.shape[0],Q + 3*W.shape[0],Q + 4*W.shape[0],Q + 5*W.shape[0], Q + 6*W.shape[0],Q + 7*W.shape[0])) # polyscope convention
        Q = np.vstack((
            np.hstack((Q + 0*W.shape[0],Q + W.shape[0], Q + 2*W.shape[0],Q + 3*W.shape[0])),
            np.hstack((Q + 4*W.shape[0],Q + 5*W.shape[0], Q + 6*W.shape[0],Q + 7*W.shape[0])),
            np.hstack((Q + 2*W.shape[0],Q + 1*W.shape[0], Q + 5*W.shape[0],Q + 6*W.shape[0])),
            np.hstack((Q + 3*W.shape[0],Q + 0*W.shape[0], Q + 4*W.shape[0],Q + 7*W.shape[0])),
            np.hstack((Q + 1*W.shape[0],Q + 0*W.shape[0], Q + 4*W.shape[0],Q + 5*W.shape[0])),
            np.hstack((Q + 3*W.shape[0],Q + 2*W.shape[0], Q + 6*W.shape[0],Q + 7*W.shape[0])),
            ))
        
    # remap faces
    V, _, SVJ, Q = remove_duplicate_vertices(V,faces=Q,epsilon=np.amin(W)/100)
    H = SVJ[H]
    return V,Q,H
