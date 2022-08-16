import numpy as np

def initialize_aabbtree(V,F=None,vmin=None,vmax=None):
    """Axis-Aligned Bounding Box hierarchy for efficient computation

    Computes an AABB tree by recursively dividing cells along the biggest dimension. May be called for any dimension and simplex size

    Parameters
    ----------
    V : numpy double array
        Matrix of mesh vertex coordinates
    F : numpy int array, optional (default None)
        Matrix of element indices into V (if None, treated as point cloud)

    Returns
    -------
    C : numpy double array 
        Matrix of box centers
    W : numpy double array 
        Matrix of box widths
    CH : numpy int array
        Matrix of child indeces (-1 if leaf node)
    PAR : numpy int array 
        Vector of immediate parent indeces (to traverse upwards)
    D : numpy int array
        Vector of tree depths
    tri_indices : numpy int array
        Vector of element indices (-1 if *not* leaf node)

    See Also
    --------
    traverse_aabbtree, initialize_quadtree.

    Notes
    -----
    This code is *purposefully* not optimized beyond asymptotics for simplicity in understanding its functionality and translating it to other programming languages beyond prototyping.

    Examples
    --------
    ```python
    from gpytoolbox import initialize_aabbtree
    P = np.array([[0,0,0],[0.1,0,0],[0,0,0],[0.02,0.1,0],[1,1,0.9],[1,1,1]])
    F = np.array([[0,1],[2,3],[4,5]],dtype=int)
    C,W,CH,PAR,D,tri_ind = gpytoolbox.initialize_aabbtree(P,F)
    ```
    """

    if (F is None):
        F = np.linspace(0,V.shape[0]-1,V.shape[0],dtype=int)[:,None]

    # We start with a bounding box
    dim = V.shape[1]
    simplex_size = F.shape[1]
    num_s = F.shape[0]
    if (vmin is None):
        vmin = np.amin(V,axis=0)
    if (vmax is None):
        vmax = np.amax(V,axis=0)
    C = (vmin + vmax)/2.0
    C = C[None,:]
    #print(C)

    # We need to compute this once, we'll need it for the subdivision:
    vmin_big = 10000*np.ones((num_s,dim))
    vmax_big = -10000*np.ones((num_s,dim))
    for j in range(dim):
        for i in range(simplex_size):
            vmax_big[:,j] = np.amax(np.hstack((V[F[:,i],j][:,None],vmax_big[:,j][:,None])),axis=1)[:]
            vmin_big[:,j] = np.amin(np.hstack((V[F[:,i],j][:,None],vmin_big[:,j][:,None])),axis=1)[:]


    W = np.reshape(vmax-vmin,(1,dim))
    CH = np.tile(np.array([[-1]],dtype=int),(1,2)) # for now it's leaf node
    D = np.array([1],dtype=int)
    PAR = np.array([-1],dtype=int) # supreme Neanderthal ancestral node
    tri_indices = np.array([-1]) # this will have the index in leaf nodes (at least temporarily)
    tri_index_list = [ list(range(F.shape[0])) ]
    # Now, we will loop
    box_ind = -1
    while True:
        box_ind = box_ind + 1
        if box_ind>=C.shape[0]:
            break
        is_child = (CH[box_ind,1]==-1)
        assert(is_child) # This check should be superfluous I think
        #tris_in_box = is_in_box(V,F,C[box_ind,:],W[box_ind,:])
        tris_in_box = tri_index_list[box_ind]
        # print(box_ind)
        # print(tris_in_box)
        # print(tri_indices)
        # print(tris_in_box)
        num_tris_in_box = len(tris_in_box)
        # print(num_tris_in_box)
        assert(num_tris_in_box>0) # There can't be a node with zero triangles...
        if (is_child and num_tris_in_box>=2):
            # Does this quad contain more than one triangle? Then subdivide it
            C,W,CH,PAR,D,tri_indices,tri_index_list = subdivide_box(box_ind,V,F,tris_in_box,C,W,CH,PAR,D,tri_indices,tri_index_list,vmin_big,vmax_big)
        if (is_child and num_tris_in_box==1):
            tri_indices[box_ind] = tris_in_box[0] # Check this??

    return C,W,CH,PAR,D,tri_indices


def subdivide_box(box_ind,V,F,tris_in_box,C,W,CH,PAR,D,tri_indices,tri_index_list,vmin_big,vmax_big):
    # First: find largest dimension
    num_tris_in_box = len(tris_in_box)
    ncp = 2 # Dimension-agnostic number of children per parent
    num_boxes = C.shape[0]
    # We will build a vector of maximums where each row is the maximum along all dimensions for each element
    vmin = vmin_big[tris_in_box,:]
    vmax = vmax_big[tris_in_box,:]


    spread = np.amax(vmax,axis=0) - np.amin(vmin,axis=0)
    best_dim = np.argmax(spread)
    # Third: pick median along best axis and separate into "left" triangles and "right" triangles
    max_values_along_best_dim = vmax[:,best_dim]
    sorted_max_values = np.argsort(max_values_along_best_dim)
    left_inds = sorted_max_values[0:((num_tris_in_box//2))]
    right_inds = sorted_max_values[((num_tris_in_box//2)):num_tris_in_box]
    # split_value = np.median(max_values_along_best_dim)+tol
    # left_inds = np.nonzero(max_values_along_best_dim<split_value)[0]
    # right_inds = np.nonzero(max_values_along_best_dim>=split_value)[0]
    # print(max_values_along_best_dim)
    # #print(split_value)
    # print(left_inds)
    # print(right_inds)
    # Find new center and widths
    vmax_l = np.amax(vmax[left_inds,:],axis=0)
    vmin_l = np.amin(vmin[left_inds,:],axis=0)
    center_left = 0.5*(vmax_l + vmin_l)
    w_left = vmax_l - vmin_l

    vmax_r = np.amax(vmax[right_inds,:],axis=0)
    vmin_r = np.amin(vmin[right_inds,:],axis=0)
    center_right = 0.5*(vmax_r + vmin_r)
    w_right = vmax_r - vmin_r

    # Add two new cells at the bottom of C, W, CH, PAR, D, tri_indeces
    C = np.vstack((
        C,
        center_left,
        center_right
    ))

    W = np.vstack((
        W,
        w_left,
        w_right
    ))
    # Add new leaf nodes
    CH = np.vstack((
        CH,
        np.tile(np.array([[-1]]),(ncp,ncp))
    ))
    # Update children of current node
    CH[box_ind,:] =  num_boxes + np.linspace(0,ncp-1,ncp,dtype=int)

    # Update parenthood:
    PAR = np.concatenate((PAR,np.tile(np.array([box_ind],dtype=int),ncp)))
    # And depth:
    D = np.concatenate((D,np.tile(np.array([D[box_ind]+1],dtype=int),ncp)))
    tri_indices = np.vstack((tri_indices,np.array([[-1],[-1]])))

    tris_in_box_left = []
    for i in range(len(left_inds)):
        tris_in_box_left.append(tris_in_box[left_inds[i]])
    tris_in_box_right = []
    for i in range(len(right_inds)):
        tris_in_box_right.append(tris_in_box[right_inds[i]])
    tri_index_list.append( tris_in_box_left )
    tri_index_list.append( tris_in_box_right )
    return C,W,CH,PAR,D,tri_indices,tri_index_list

def is_in_box(V,F,center,width):
    # Checks if triangle is FULLY contained in box
    # To do
    tol = 1e-8
    dim = V.shape[1]
    num_faces = F.shape[0]
    simplex_size = F.shape[1]
    vmin = 10000*np.ones((num_faces,dim))
    vmax = -10000*np.ones((num_faces,dim))
    for j in range(dim):
        for i in range(simplex_size):
            vmax[:,j] = np.amax(np.hstack((V[F[:,i],j][:,None],vmax[:,j][:,None])),axis=1)
            vmin[:,j] = np.amin(np.hstack((V[F[:,i],j][:,None],vmin[:,j][:,None])),axis=1)
    # print(vmax)
    # print(vmin)
    # print(center)
    # print(width)
    is_in_box_dim = np.zeros((num_faces,dim),dtype=np.bool_)
    for i in range(dim):
        is_in_box_dim[:,i] = np.logical_and((vmin[:,i] >=  (center[i]-0.5*width[i] - tol) ),(vmax[:,i] <=  (center[i]+0.5*width[i] + tol) ) )
    is_in_box_vec = np.all(is_in_box_dim,axis=1)
    return is_in_box_vec