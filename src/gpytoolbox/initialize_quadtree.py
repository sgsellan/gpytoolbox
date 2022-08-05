import numpy as np
from scipy.sparse import csr_matrix
from . subdivide_quad import subdivide_quad




def initialize_quadtree(P,max_depth=8,min_depth=1,graded=False,vmin=None,vmax=None):
    """Builds quadtree or octree from given point cloud.

    Builds an adaptatively refined (optionally graded) quadtree for prototyping on adaptative grids. Keeps track of all parenthood and adjacency information so that traversals and differential quantities are easy to compute. 

    Parameters
    ----------
    P : numpy double array
        Matrix of point cloud coordinates
    max_depth : int, optional (default 8)
        max tree depth (min edge length will be bounding_box_length*2^(-max_depth))
    min_depth : int, optional (default 1)
        minimum tree depth (depth one is a single box)
    graded bool 
        Whether to ensure that adjacent quads only differ by one in depth or not (this is useful for numerical applications, not so much for others like position queries).

    Returns
    -------
    C : numpy double array 
        Matrix of cell centers
    W : numpy double array 
        Vector of cell half widths
    CH : numpy int array
        Matrix of child indeces (-1 if leaf node)
    PAR : numpy int array 
        Vector of immediate parent indeces (to traverse upwards)
    D : numpy int array
        Vector of tree depths
    A : scipy sparse.csr_matrix
        Sparse node adjacency matrix, where a value of a in the (i,j) entry means that node j is to the a-th direction of i (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).

    See Also
    --------
    quadtree_children, in_quadtree.

    Notes
    -----
    This code is *purposefully* not optimized beyond asymptotics for simplicity in understanding its functionality and translating it to other programming languages beyond prototyping.

    Examples
    --------
    TO-DO
    """

    # We start with a bounding box
    dim = P.shape[1]
    if (vmin is None):
        vmin = np.amin(P,axis=0)
    if (vmax is None):
        vmax = np.amax(P,axis=0)
    C = (vmin + vmax)/2.0
    C = C[None,:]
    #print(C)
    W = np.array([np.amax(vmax-vmin)])
    CH = np.tile(np.array([[-1]],dtype=int),(1,2**dim)) # for now it's leaf node
    D = np.array([1],dtype=int)
    A = csr_matrix((1,1))
    PAR = np.array([-1],dtype=int) # supreme Neanderthal ancestral node


    # Now, we will loop
    quad_ind = -1
    while True:
        quad_ind = quad_ind + 1
        if quad_ind>=C.shape[0]:
            break
        is_child = (CH[quad_ind,1]==-1)
        # Does this quad contain any point? (Or is it below our min depth)
        if ((D[quad_ind]<min_depth or np.any(is_in_quad(P,C[quad_ind,:],W[quad_ind]))) and D[quad_ind]<max_depth and is_child):
            # If it does, subdivide it
            C,W,CH,PAR,D,A = subdivide_quad(quad_ind,C,W,CH,PAR,D,A,graded)
    return C,W,CH,PAR,D,A




# This just checks if a point is in a square
def is_in_quad(queries,center,width):
    dim = queries.shape[1]
    max_corner = center + width*np.tile(np.array([0.5]),dim)
    min_corner = center - width*np.tile(np.array([0.5]),dim)
    b = np.ones(queries.shape[0],dtype=bool)
    for dd in range(dim):
        b = (b & (queries[:,dd]>=min_corner[dd]) & (queries[:,dd]<=max_corner[dd]))
    return b