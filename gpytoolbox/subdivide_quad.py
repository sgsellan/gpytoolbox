import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack

def subdivide_quad(ind,C1,W1,CH1,PAR1,D1,A1,graded1):
    # Subdivides the ind-th cell of a given octree, maintaining all the #
    # adjacency information
    #
    # Inputs:
    #   ind integer index of cell to subdivide
    #   C1 #nodes by 3 matrix of cell centers
    #   W1 #nodes vector of cell widths (**not** half widths)
    #   CH1 #nodes by 4 matrix of child indeces (-1 if leaf node)
    #   PAR1 #nodes vector of immediate parent indeces (to traverse upwards)
    #   D1 #nodes vector of tree depths
    #   A1 #nodes by #nodes sparse adjacency matrix, where a value of a in the
    #       (i,j) entry means that node j is to the a-th direction of i
    #       (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).
    #  Optional:
    #       graded1 boolean whether to ensure that adjacent quads only differ by
    #           one in depth or not (this is useful for numerical applications, 
    #           not so much for others like position queries).
    #
    # Outputs:
    #   C2 #nodes by 3 matrix of cell centers
    #   W2 #nodes vector of cell widths (**not** half widths)
    #   CH2 #nodes by 4 matrix of child indeces (-1 if leaf node)
    #   PAR2 #nodes vector of immediate parent indeces (to traverse upwards)
    #   D2 #nodes vector of tree depths
    #   A2 #nodes by #nodes sparse adjacency matrix, where a value of a in the
    #       (i,j) entry means that node j is to the a-th direction of i
    #       (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).
    #
    assert(CH1[ind,1]==-1) # can't subdivide if not a leaf node
    # For simplicity:
    w = W1[ind]
    c = C1[ind,:]
    d = D1[ind]
    p = PAR1[ind]
    a_ind = A1[:,ind]
    num_quads = C1.shape[0]
    # Easy part: add four cells new cell centers order: bottom-left,
    # bottom-right, top-left, top-right
    C2 = np.vstack(
        (
            C1,
            c[None,:] + 0.25*w*np.array([[-1,-1]]),
            c[None,:] + 0.25*w*np.array([[1,-1]]),
            c[None,:] + 0.25*w*np.array([[-1,1]]),
            c[None,:] + 0.25*w*np.array([[1,1]])
        )
     )
    # New widths
    W2 = np.concatenate((W1,np.array([0.5*w,0.5*w,0.5*w,0.5*w])))
    # New depths
    D2 = np.concatenate((D1,np.array([d+1,d+1,d+1,d+1],dtype=int)))
    # Keep track of child indeces
    CH2 = np.vstack((
        CH1,
        np.tile(np.array([[-1,-1,-1,-1]]),(4,1))
    ))
    CH2[ind,:] = num_quads + np.array([0,1,2,3],dtype=int)
    # And parent indeces
    PAR2 = np.concatenate((PAR1,np.array([ind,ind,ind,ind],dtype=int)))
    # Now the hard part, which is the adjacency Reminder:
    # (left-right-bottom-top) Effectively we are concatenating [A , B;  "-B", C]
    # C is always the same square 4 by 4 matrix
    square_mat = csr_matrix(np.array([
        [0,2,4,0],
        [1,0,0,4],
        [3,0,0,2],
        [0,3,1,0]
    ]))
    rect_mat = csr_matrix((num_quads,4))
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
            J = np.array([neighbor_ind,neighbor_ind])
            # Orientation will also be the same that it was
            vals = np.array([neighbor_where,neighbor_where])
            # The tricky bit is *which* are the new neighbors order:
            # bottom-left, bottom-right, top-left, top-right
            if neighbor_where==1: # if ind_quad is to the left of neighbor_ind
                I = np.array([1,3]) # right indeces
            elif neighbor_where==2: # if ind_quad is to the right of neighbor_ind
                I = np.array([0,2]) # left indeces
            elif neighbor_where==3: # if ind_quad is to the bottom of neighbor_ind
                I = np.array([2,3]) # top indeces
            elif neighbor_where==4: # if ind_quad is to the top of neighbor_ind
                I = np.array([0,1]) # bottom indeces
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
            if which_child==0: # it comes from the bottom left bit of the depth-d neighbor
                # Then there are two options, either ind_quad is to its left or
                # to its bottom
                if neighbor_where==1: # if ind_quad is to the left of neighbor_ind
                    I = np.array([1]) # then this is the bottom right of neighbor_ind
                elif neighbor_where==3: # if ind_quad is to the bottom of neighbor_ind
                    I = np.array([2]) # then this is the top left
            elif which_child==1: # it comes from the BOTTOM RIGHT bit of the depth-d neighbor
                # Then there are two options, either ind_quad is to its right or
                # to its bottom
                if neighbor_where==2: # right
                    I = np.array([0]) # bottom left
                elif neighbor_where==3: # bottom
                    I = np.array([3]) # top right
            elif which_child==2: # it comes from the TOP LEFT bit of the depth-d neighbor
                # two option: top or left
                if neighbor_where==1: # left
                    I = np.array([3]) # top right
                elif neighbor_where==4: # top
                    I = np.array([0]) # bottom left
            elif which_child==3: # it comes from the TOP RIGHT bit
                # two options: top or right
                if neighbor_where==4: # top
                    I = np.array([1]) # bottom right
                elif neighbor_where==2:
                    I = np.array([2])
            # ...phew. Again, that should be a lookup table
        rect_mat = rect_mat + csr_matrix((vals,(J,I)),shape=(num_quads,4))
    A2 = csr_matrix(vstack((
        hstack((A1,rect_mat)),
        hstack((transpose_orientation(rect_mat),square_mat))
    )))
    #print("A2") print(A2.toarray())
    if graded1:
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
    return R