import numpy as np
# Bindings using Eigen and libigl:
import sys
import os
from .remove_duplicate_vertices import remove_duplicate_vertices


def bad_quad_mesh_from_quadtree(C,W,CH):
    """Builds a mesh of a quadtree or octree for visualization purposes.

    From a proper quadtree, builds a vertex-connected but degenerate quad mesh
    containing only the leaf nodes, to be used for visualizing the quadtree and quantities defined on its leaf nodes.

    Parameters
    ----------
    C : numpy double array
        Matrix of cell centers
    W : numpy double array
        Vector of half cell widths
    CH : numpy int array
        Matrix of child indices (-1 if leaf node)

    Returns
    -------
    V : numpy double array
        Matrix of mesh vertices
    Q : numpy int array
        Matrix of quad mesh indices
    H : numpy int array
        Matrix of hexahedral mesh indices if input is octree (empty if quadtree)

    See Also
    --------
    initialize_quadtree, quadtree_children.

    Examples
    --------
    TO-DO
    """
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
