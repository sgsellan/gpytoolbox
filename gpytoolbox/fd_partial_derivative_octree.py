import numpy as np
from numpy.core.function_base import linspace
from scipy.sparse import csr_matrix


def fd_partial_derivative_octree(V,Q,direction=0):
    # WE ASSUME Q IS GIVEN IN THE POLYSCOPE ORDERING!!
    # Du stores derivative values at staggered octree locations; i.e. in the middle of each edge...
    # This should output these staggered positions

    # since this is an octree all edges of each cell will be equal
    edge_lengths_cell = np.linalg.norm(V[Q[:,0],:]-V[Q[:,1],:],axis=1)[:,None]
    # Each cell will have four entries in this derivative:
    unit_vec = np.array([0,0,0])
    unit_vec[direction] = 1
    if direction==0:
        edges = [[0,3],[1,2],[4,7],[5,6]]
    elif direction==1:
        edges = [[0,1],[3,2],[4,5],[7,6]]
    elif direction==2:
        edges = [[0,4],[1,5],[2,6],[3,7]]
    
    E = np.vstack((Q[:,edges[0]],Q[:,edges[1]],Q[:,edges[2]],Q[:,edges[3]]))
    edge_lengths = np.vstack((edge_lengths_cell,edge_lengths_cell,edge_lengths_cell,edge_lengths_cell))
    I2 = np.tile(np.linspace(0,E.shape[0]-1,E.shape[0],dtype=int)[:,None],(1,2))
    vals2 = np.hstack((-edge_lengths,edge_lengths))
    staggered = (V[E[:,0],:] + V[E[:,1],:])/2
    # print(vals2)
    J = np.reshape(E,(-1,1),order='F').squeeze()
    I = np.reshape(I2,(-1,1),order='F').squeeze()
    vals = 1/np.reshape(vals2,(-1,1),order='F').squeeze()
    # print(I)
    # print(J)
    # print(vals)
    D = csr_matrix((vals,(I,J)),shape=(E.shape[0],V.shape[0]))


    return D,staggered