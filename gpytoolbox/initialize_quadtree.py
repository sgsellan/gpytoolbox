import numpy as np
from scipy.sparse import csr_matrix
from . subdivide_quad import subdivide_quad




def initialize_quadtree(P,max_depth=8,min_depth=1,graded=False,vmin=None,vmax=None):
    # Builds an adaptatively refined (optionally graded) quadtree for
    # prototyping on adaptative grids. Keeps track of all parenthood and
    # adjacency information so that traversals and differential quantities are
    # easy to compute. This code is *purposefully* not optimized beyond
    # asymptotics for simplicity in understanding its functionality and
    # translating it to other programming languages beyond prototyping.
    #
    #
    #
    # Inputs:
    #   P is a #P by 3 matrix of points. The output tree will be more subdivided in
    #       regions with more points
    #   Optional:
    #       MinDepth integer minimum tree depth (depth one is a single box)
    #       MaxDepth integer max tree depth (min edge length will be
    #           bounding_box_length*2^(-MaxDepth))
    #       Graded boolean whether to ensure that adjacent quads only differ by
    #           one in depth or not (this is useful for numerical applications, 
    #           not so much for others like position queries).
    #
    # Outputs:
    #   C #nodes by 3 matrix of cell centers
    #   W #nodes vector of cell widths (**not** half widths)
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #   PAR #nodes vector of immediate parent indeces (to traverse upwards)
    #   D #nodes vector of tree depths
    #   A #nodes by #nodes sparse adjacency matrix, where a value of a in the
    #       (i,j) entry means that node j is to the a-th direction of i
    #       (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).
    #



    # We start with a bounding box
    if (vmin is None):
        vmin = np.amin(P,axis=0)
    if (vmax is None):
        vmax = np.amax(P,axis=0)
    C = (vmin + vmax)/2.0
    C = C[None,:]
    #print(C)
    W = np.array([np.amax(vmax-vmin)])
    CH = np.array([[-1,-1,-1,-1]],dtype=int) # for now it's leaf node
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
    max_corner = center + width*np.array([0.5,0.5])
    min_corner = center - width*np.array([0.5,0.5])
    return ( (queries[:,0]>=min_corner[0]) & (queries[:,1]>=min_corner[1])    & (queries[:,0]<=max_corner[0]) & (queries[:,1]<=max_corner[1]) )