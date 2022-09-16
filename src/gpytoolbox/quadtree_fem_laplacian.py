import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy import integrate
from .compactly_supported_normal import compactly_supported_normal


def quadtree_fem_laplacian(C,W,CH,D,A):
    # Builds a finite element laplacian 
    #
    # G = quadtree_laplacian(C,W,CH,D,A)
    # G,stored_at = quadtree_laplacian(C,W,CH,D,A)
    #
    # Inputs:
    #   C #nodes by 3 matrix of cell centers
    #   W #nodes vector of cell widths (**not** half widths)
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #   D #nodes vector of tree depths
    #   A #nodes by #nodes sparse adjacency matrix, where a value of a in the
    #       (i,j) entry means that node j is to the a-th direction of i
    #       (a=1: left  a=2: right  a=3: bottom  a=4: top).
    #
    # Outputs:
    #   G #num_children by #num_children sparse laplacian matrix 
    #   stored_at #num_children by 3 matrix of child cell centers, where the
    #       values of L are stored

    dim = C.shape[1]
    num_faces_per_cell = 2*dim
    
    # For debugging, this numerically computes the entries
    def lij_slow(ii,jj):
        # sample_point = C[ii,:]-0.001
        #print(ii,jj)
        #print(ii,jj)
        # liijj = np.linalg.norm(C[ii,:]-C[jj,:])
        # liijj = compactly_supported_normal(C[jj,:][None,:],n=4,sigma=W[ii],center=C[ii,:])
        liijj = 0
        for dd in range(dim):
            # We add <d^2F_i/dx_i^2,F_j>
            def func1(x,y):
                return compactly_supported_normal(np.array([[x,y]]),n=2,sigma=W[ii],center=C[ii,:],second_derivative=dd)
            def func2(x,y):
                return compactly_supported_normal(np.array([[x,y]]),n=2,sigma=W[jj],center=C[jj,:],second_derivative=-1)
            def int_func(x,y):
                return func1(x,y)*func2(x,y).item()
            int_val,_ = integrate.nquad(int_func, [[C[ii,0]-1.5*W[ii],C[ii,0]+1.5*W[ii]],[C[ii,1]-1.5*W[ii],C[ii,1]+1.5*W[ii]]] )
            liijj = liijj + int_val
            #print(dd,int_val)
            #print(func2(sample_point[0],sample_point[1]))
        return liijj
    
    def lij(ii,jj):
        #sample_point = C[ii,:]-0.001
        # print(ii,jj)
        # liijj = np.linalg.norm(C[ii,:]-C[jj,:])
        # liijj = compactly_supported_normal(C[jj,:][None,:],n=4,sigma=W[ii],center=C[ii,:])
        liijj = 0
        for dd in range(dim):
            int_val = 1
            debug_val = 1
            # We add <d^2F_i/dx_i^2,F_j>
            # which is the product of int(f_i*f_j) or int(d^2f_i*f_j)
            for dd_int in range(dim):
                def func1(x):
                    if dd_int==dd:
                        return compactly_supported_normal(np.reshape(x,(-1,1)),n=2,sigma=W[ii],center=np.array([C[ii,dd_int]]),second_derivative=0)
                    else:
                        return compactly_supported_normal(np.reshape(x,(-1,1)),n=2,sigma=W[ii],center=np.array([C[ii,dd_int]]),second_derivative=-1)
                def func2(x):
                    return compactly_supported_normal(np.reshape(x,(-1,1)),n=2,sigma=W[jj],center=np.array([C[jj,dd_int]]),second_derivative=-1)
                def int_func(x):
                    #print(x)
                    #print(func1(x))
                    return func1(x)*func2(x)
                #val,_ = integrate.nquad(int_func, [[-1,1]] )
                # To do this *exactly*, we should break it into the intervals given by 

                interval_endpoints = [C[ii,dd_int] - 1.5*W[ii], 
                C[ii,dd_int] - 0.5*W[ii],
                C[ii,dd_int] + 0.5*W[ii],
                C[ii,dd_int] + 1.5*W[ii],
                C[jj,dd_int] - 1.5*W[jj],
                C[jj,dd_int] - 0.5*W[jj],
                C[jj,dd_int] + 0.5*W[jj],
                C[jj,dd_int] + 1.5*W[jj]]
                interval_endpoints.sort()
                running_integral = 0 # we will add each interval's contribution here
                for interval_ind in range(len(interval_endpoints)-1):
                    val,_ = integrate.fixed_quad(int_func,interval_endpoints[interval_ind],interval_endpoints[interval_ind+1],n=3)
                    running_integral = running_integral + val
                val = running_integral
                #val,_ = integrate.fixed_quad(int_func,C[ii,dd_int]-1.5*W[ii],C[ii,dd_int]+1.5*W[ii],n=50)
                int_val = int_val*val
                #debug_val = debug_val*func2(sample_point[dd_int])
            liijj = liijj + int_val
            #print(debug_val)
        return liijj
    
    
    # We will store Laplacian values at
    # child cell indeces
    children = np.nonzero(CH[:,1]==-1)[0]
    # map from all cells to children
    cell_to_children = -np.ones(W.shape[0],dtype=int)
    cell_to_children[children] = np.linspace(0,children.shape[0]-1,children.shape[0],dtype=int)

    # Vectors for constructing the Laplacian
    I = []
    J = []
    vals = []
    

    for i in range(children.shape[0]):
        new_I = []
        new_J = []
        new_vals = []
        child_i = children[i]
        # TODO: Change this to loop only over possible nonzero entries
        for j in range(children.shape[0]):
            child_j = children[j]
            new_I.append(i)
            new_J.append(j)
            #print(lij(child_i,child_j),lij_2(child_i,child_j))
            new_vals.append(lij(child_i,child_j))
            
        I.extend(new_I)
        J.extend(new_J)
        vals.extend(new_vals)
    

    

    # print(I)
    # print(J)
    # print(vals)
    # THE LAPLACIAN IS NEGATIVE SEMI DEFINITE!
    L = csr_matrix((vals,(I,J)),(children.shape[0],children.shape[0]))
    # L = L - diags(np.array(L.sum(axis=1)).squeeze(),0)
    stored_at = C[children,:]
    return L, stored_at
    