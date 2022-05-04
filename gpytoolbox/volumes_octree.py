import numpy as np
from scipy.sparse import csr_matrix


def volumes_octree(V,Q):
    # WE ASSUME Q IS GIVEN IN THE POLYSCOPE ORDERING!!

    # since this is an octree all edges of each cell will be equal
    vols = np.linalg.norm(V[Q[:,0],:]-V[Q[:,1],:],axis=1)**3.0
    I = np.linspace(0,Q.shape[0]-1,Q.shape[0],dtype=int)
    A = csr_matrix((vols,(I,I)),shape=(Q.shape[0],Q.shape[0]))
    return A