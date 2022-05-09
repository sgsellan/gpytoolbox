import numpy as np
from scipy.sparse import csr_matrix, diags

def quadtree_laplacian(C,W,CH,D,A):
    # Builds a finite difference laplacian on a quadtree following a centered 
    # finite difference scheme, with the adjacency as suggested by 
    # Bickel et al. "Adaptative Simulation of Electrical
    # Discharges". This code is *purposefully* not optimized beyond
    # asymptotics for simplicity in understanding its functionality and
    # translating it to other programming languages beyond prototyping.
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
        l = [1,1,1,1,1]
        new_dirs = []
        child = children[i]
        d = D[child]
        num_dirs = 0
        # Let's build d u(child)/dx^2 ~ u(child+W(child)*[1,0])/hr(hl+hr) -
        # 2u(child)/hlhr + u(child-W(child)*[1,0])/hr(hl+hr)
        # So, let's look for the value to the j direction. To do this, we seek the
        # lowest-depth neighbor to the j direction. As a reminder the octree
        # adjacency convention is i->j (1:left-2:right-3:bottom-4:top)
        for j in range(1,5):
            j_neighbors = (A[child,:]==j).nonzero()[1]
            if len(j_neighbors)>0:
                depths_j_neighbors = D[j_neighbors]
                max_depth_j_neighbor = np.argmax(depths_j_neighbors)
                max_depth_j = depths_j_neighbors[max_depth_j_neighbor]
                max_depth_j_neighbor = j_neighbors[max_depth_j_neighbor]
                # There are two options:
                # One: the leaf node to our j direction has lower or equal depth to
                # us
                if max_depth_j<=d:
                    l[j] = (W[child] + W[max_depth_j_neighbor])/2.0
                    # then it's easy, just add this node
                    new_I.append(i)
                    # THIS HAS TO BE A CHILD !
                    assert(cell_to_children[max_depth_j_neighbor]>=0)
                    new_J.append(cell_to_children[max_depth_j_neighbor])
                    new_vals.append(-1.0)
                    new_dirs.append(j)
                else:
                    # In this case, assuming the grid is graded, there should
                    # be two j-neighbors at depth d+1
                    nn = j_neighbors[D[j_neighbors]==(d+1)]
                    assert len(nn)==2, "Are you sure you are inputting a graded quadtree?"
                    assert all(CH[nn,1]==-1)
                    # Then we simply average both
                    l[j] = (W[child] + W[nn[1]])/2.0
                    new_I.append(i)
                    new_I.append(i)
                    new_J.append(cell_to_children[nn[0]])
                    new_J.append(cell_to_children[nn[1]])
                    new_vals.append(-0.5)
                    new_vals.append(-0.5)
                    new_dirs.append(j)
                    new_dirs.append(j)
                
                num_dirs = num_dirs + 1
            
        
            
        # At this point, we have to divide by the edge-lengths and add sign
        for s in range(len(new_dirs)):
            if new_dirs[s]==1:
                new_vals[s] = new_vals[s]/(l[1]*(l[1]+l[2]))
            elif new_dirs[s]==2:
                new_vals[s] = new_vals[s]/(l[2]*(l[1]+l[2]))
            elif new_dirs[s]==3:
                new_vals[s] = new_vals[s]/(l[3]*(l[3]+l[4]))
            elif new_dirs[s]==4:
                new_vals[s] = new_vals[s]/(l[4]*(l[3]+l[4]))
    
        # And add them to the big sparse Laplacian construction vectors
        I.extend(new_I)
        J.extend(new_J)
        vals.extend(new_vals)
    
    
    # THE LAPLACIAN IS NEGATIVE SEMI DEFINITE!
    L = -2*csr_matrix((vals,(I,J)),(children.shape[0],children.shape[0]))
    L = L - diags(np.array(L.sum(axis=1)).squeeze(),0)
    stored_at = C[children,:]
    return L, stored_at
    
    