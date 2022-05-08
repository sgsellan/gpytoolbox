import numpy as np
import igl

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
    #
    # Example:
    #
    #
    # See also: initialize_quadtree

    is_child = (CH[:,1]==-1)
    W = W[is_child][:,None]
    C = C[is_child,:]
    # translate this
    V = np.vstack((
        C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[-1,-1]]),(W.shape[0],1)),
        C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[-1,1]]),(W.shape[0],1)),
        C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[1,1]]),(W.shape[0],1)),
        C + 0.5*np.tile(W,(1,2))*np.tile(np.array([[1,-1]]),(W.shape[0],1)),
        ))
    Q = np.linspace(0,W.shape[0]-1,W.shape[0],dtype=int)[:,None]
    Q = np.hstack((
        Q,
        Q + W.shape[0],
        Q + 2*W.shape[0],
        Q + 3*W.shape[0]
    ))

    # remap faces
    V, _, _, Q = igl.remove_duplicate_vertices(V,Q,np.amin(W)/100)
    return V,Q
