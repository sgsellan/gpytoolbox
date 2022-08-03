import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, eye

def subdivide_quad(ind,C1,W1,CH1,PAR1,D1,A1,graded=True):
    """"Turn one cell of a quadtree or octree into four or eight, respectively
    
    Subdivides the ind-th cell of a given octree, maintaining all the adjacency information

    Parameters
    ----------
    ind : int 
        Index of cell to subdivide into input tree
    C1 : numpy double array 
        Matrix of cell centers of input tree
    W1 : numpy double array 
        Vector of cell half widths of input tree
    CH1 : numpy int array
        Matrix of child indeces (-1 if leaf node) of input tree
    PAR1 : numpy int array 
        Vector of immediate parent indeces (to traverse upwards) of input tree
    D1 : numpy int array
        Vector of tree depths of input tree
    A1 : scipy sparse.csr_matrix
        Sparse node adjacency matrix of input tree, where a value of a in the (i,j) entry means that node j is to the a-th direction of i (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).
    graded : bool, optional (default True)
        Whether to ensure that adjacent quads only differ by one in depth or not (this is useful for numerical applications, not so much for others like position queries).

    Returns
    -------
    C2 : numpy double array 
        Matrix of cell centers of output tree
    W2 : numpy double array 
        Vector of cell half widths of output tree
    CH2 : numpy int array
        Matrix of child indeces (-1 if leaf node) of output tree
    PAR2 : numpy int array 
        Vector of immediate parent indeces (to traverse upwards) of output tree
    D2 : numpy int array
        Vector of tree depths of output tree
    A2 : scipy sparse.csr_matrix
        Sparse node adjacency matrix of output tree, where a value of a in the (i,j) entry means that node j is to the a-th direction of i (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).

    See Also
    --------
    initialize_quadtree.

    Examples
    --------
    TODO
    """

    # If quad A lays to the <lookup[:,1]> of quad B, then children
    # <lookup[:,2:end]> resulting from subdividing A will still be
    # neighbors of quad B
    # legend for <lookup[:,1]> is left, right, bottom, top, (back, front)
    # children legend is 1: back bottom left, 2: back bottom right, 3: back top
    # left, 4: back top right, 5: front bottom left, 6: front bottom right, 7: front
    # top left, 8: front top right
    lookup_a = np.array([[1,2,4,8,6], # A to the left of B, so we add all right children of A
                        [2,1,3,5,7],
                        [3,3,4,7,8],
                        [4,1,2,5,6],
                        [5,5,6,7,8],
                        [6,1,2,3,4]],dtype=int) 
                        
    # (reminder 2:end columns are 1-indexed)
    lookup_a[:,1:5] = lookup_a[:,1:5] - 1
    
    # If quad A lays to the <lookup_b[:,1]> of quad B, and we subdivide both,
    # then child <lookup_b[:,2]> of quad B will have child <lookup_b[:,3]> of
    # quad A as a new neighbor:
    lookup_b = np.array([[1,1,2],
                        [1,3,4],
                        [1,5,6],
                        [1,7,8],
                        [2,2,1],
                        [2,4,3],
                        [2,6,5],
                        [2,8,7],
                        [3,1,3],
                        [3,2,4],
                        [3,5,7],
                        [3,6,8],
                        [4,3,1],
                        [4,4,2],
                        [4,7,5],
                        [4,8,6],
                        [5,3,7],
                        [5,4,8],
                        [5,1,5],
                        [5,2,6],
                        [6,7,3],
                        [6,8,4],
                        [6,5,1],
                        [6,6,2]],dtype=int) 
                        
    # (reminder 2:end columns are 1-indexed)
    lookup_b[:,1:3] = lookup_b[:,1:3] - 1





    assert(CH1[ind,1]==-1) # can't subdivide if not a leaf node
    # For simplicity:
    dim1 = C1.shape[1]
    ncp = 2**dim1 #number of children per parent for dim-agnostic
    w = W1[ind]
    c = C1[ind,:]
    d = D1[ind]
    p = PAR1[ind]
    a_ind = A1[:,ind]
    num_quads = C1.shape[0]
    # Easy part: add four cells new cell centers order: bottom-left,
    # bottom-right, top-left, top-right
    if dim1==2:
        C2 = np.vstack(
            (
                C1,
                c[None,:] + 0.25*w*np.array([[-1,-1]]),
                c[None,:] + 0.25*w*np.array([[1,-1]]),
                c[None,:] + 0.25*w*np.array([[-1,1]]),
                c[None,:] + 0.25*w*np.array([[1,1]])
            )
        )
    else:
        C2 = np.vstack(
            (
                C1,
                c[None,:] + 0.25*w*np.array([[-1,-1,-1]]),
                c[None,:] + 0.25*w*np.array([[1,-1,-1]]),
                c[None,:] + 0.25*w*np.array([[-1,1,-1]]),
                c[None,:] + 0.25*w*np.array([[1,1,-1]]),
                c[None,:] + 0.25*w*np.array([[-1,-1,1]]),
                c[None,:] + 0.25*w*np.array([[1,-1,1]]),
                c[None,:] + 0.25*w*np.array([[-1,1,1]]),
                c[None,:] + 0.25*w*np.array([[1,1,1]])
            )
        )
    
    # New widths
    W2 = np.concatenate((W1,np.tile(np.array([0.5*w]),ncp)))
    # New depths
    D2 = np.concatenate((D1,np.tile(np.array([d+1],dtype=int),ncp)))
    # Keep track of child indeces
    CH2 = np.vstack((
        CH1,
        np.tile(np.array([[-1]]),(ncp,ncp))
    ))
    CH2[ind,:] = num_quads + np.linspace(0,ncp-1,ncp,dtype=int)
    #np.array([0,1,2,3],dtype=int)
    # And parent indeces
    PAR2 = np.concatenate((PAR1,np.tile(np.array([ind],dtype=int),ncp)))
    # Now the hard part, which is the adjacency Reminder:
    # (left-right-bottom-top) Effectively we are concatenating [A , B;  "-B", C]
    # C is always the same square 4 by 4 matrix
    square_mat = csr_matrix(np.array([
        [0,2,4,0],
        [1,0,0,4],
        [3,0,0,2],
        [0,3,1,0]
    ]))
    if dim1==3:
        square_mat = vstack((
            hstack((square_mat,6*eye(4))),
            hstack((5*eye(4),square_mat))
        ))
    rect_mat = csr_matrix((num_quads,ncp))
    # We will loop over all neighbors of ind print("A1") print(A1.toarray())
    neighbors_ind = a_ind.nonzero()[0]
    # print("a_ind") print(a_ind) print("neighbors_ind") print(neighbors_ind)
    where_neighbors = a_ind[neighbors_ind].toarray().squeeze()
    # print("where") print(where_neighbors)
    for i in range(len(neighbors_ind)):
        neighbor_ind = neighbors_ind[i]
        neighbor_where = where_neighbors[i]
        neighbor_depth = D1[neighbor_ind]
        # This will help us populate rect_mat(:,neighbor_ind) We'll build I, J
        # and val. There are two options here: if the neighbor has the same or
        # higher depth, it will gain two new neighbors. If not, it will gain
        # only one. Maybe there's a way to combine both options but for now
        # let's do it one at a time.

        # Let's start with the easy case: neighbor depth is low
        if neighbor_depth<=d:
            # J will always be the same (neighbor_ind will gain two neighbors)
            num_new_nb = 2**(dim1-1)
            J = np.tile(np.array([neighbor_ind]),num_new_nb)
            # Orientation will also be the same that it was
            vals = np.tile(np.array([neighbor_where]),num_new_nb)

            I = lookup_a[lookup_a[:,0]==neighbor_where,1:(num_new_nb+1)].squeeze()
        else:
            # neighbor depth is high. We need to traverse the tree to find out
            # which of the four d + 1 depth children this neighbor comes from
            # originally. This, combined with the neighbor_where information,
            # will tell us which one new neighbor this cell has. Should use a
            # look-up table with 16 possibilities J will always be the same
            # (neighbor_ind will gain 1 neighbor)
            J = np.array([neighbor_ind])
            vals = np.array([neighbor_where])
            #
            # The key question is which of the four depth d+1 children do we
            # originally come from. TRAVERSE!
            n_ind = neighbor_ind
            n_depth = neighbor_depth
            n_par = PAR1[neighbor_ind]
            while n_depth>(d+1):
                n_ind = n_par
                assert(n_depth==(D1[n_ind]+1))
                n_depth = D1[n_ind]
                n_par = PAR1[n_ind]
            which_child = np.nonzero(CH1[n_par,:]==n_ind)[0]
            # print("which_child") print(which_child) print(neighbor_where)
            # print(n_ind) print(CH1[n_par,:])
            assert(d==D1[n_par])
            # Reminder: bottom-left, bottom-right, top-left, top-right

            lookup_row = ((lookup_b[:,0]==neighbor_where) & (lookup_b[:,1]==which_child))
            I = lookup_b[lookup_row,2]
        rect_mat = rect_mat + csr_matrix((vals,(J,I)),shape=(num_quads,ncp))
    A2 = csr_matrix(vstack((
        hstack((A1,rect_mat)),
        hstack((transpose_orientation(rect_mat),square_mat))
    )))
    #print("A2") print(A2.toarray())
    if graded:
        a2_ind = A2[:,ind] # from the perspective of the neighbors
        # Go over all neighbors of the original quad we subdivided
        neighbors_ind = a2_ind.nonzero()[0]
        for i in range(len(neighbors_ind)):
            is_child_2 = ((CH2[neighbors_ind[i],1]==-1))
            neighbor_depth = D2[neighbors_ind[i]]
            if (is_child_2 and neighbor_depth<d):
                C2,W2,CH2,PAR2,D2,A2 = subdivide_quad(neighbors_ind[i],C2,W2,CH2,PAR2,D2,A2,True)
    return C2,W2,CH2,PAR2,D2,A2




def transpose_orientation(L):
    R = L.transpose().copy()
    # Move left to right
    R[L.transpose()==1] = 2
    R[L.transpose()==2] = 1
    # Move top to bottom
    R[L.transpose()==3] = 4
    R[L.transpose()==4] = 3
    # front to back
    R[L.transpose()==5] = 6
    R[L.transpose()==6] = 5
    return R