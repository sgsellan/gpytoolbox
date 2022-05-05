import numpy as np
import igl
import os
import sys
from scipy.sparse import csr_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build-linux/')))
from gpytoolbox_eigen_bindings import in_element_aabb

def interpolate_octree(queries,V,Q):
    # THIS ASSUMES Q IS GIVEN IN THE POLYSCOPE HEX MESH ORDERING
    T = np.vstack((Q[:,[0,3,1,5]],Q[:,[3,2,1,5]],Q[:,[0,4,5,3]],
                   Q[:,[4,5,7,3]],Q[:,[3,5,7,6]],Q[:,[2,3,5,6]]))
    I = in_element_aabb(queries,V,T)
    B = igl.barycentric_coordinates_tet(queries,V[T[I,0],:],V[T[I,1],:],V[T[I,2],:],V[T[I,3],:])
    vals = np.reshape(B,(-1,1),order='F').squeeze()
    J = np.concatenate((T[I,0],T[I,1],T[I,2],T[I,3]))
    I = np.linspace(0,queries.shape[0]-1,queries.shape[0],dtype=int)
    I = np.concatenate((I,I,I,I))
    D = csr_matrix((vals,(I,J)),shape=(queries.shape[0],V.shape[0]))

    return D