import numpy as np
from scipy.sparse import csr_matrix


def massmatrix_octree(V,Q):
    # WE ASSUME Q IS GIVEN IN THE POLYSCOPE ORDERING!!

    # since this is an octree all edges of each cell will be equal
    cell_volumes = np.linalg.norm(V[Q[:,0],:]-V[Q[:,1],:],axis=1)**3.0
    vals = np.concatenate((cell_volumes,cell_volumes,cell_volumes,cell_volumes,cell_volumes,cell_volumes,cell_volumes,cell_volumes))/8.0
    I = np.concatenate((Q[:,0],Q[:,1],Q[:,2],Q[:,3],Q[:,4],Q[:,5],Q[:,6],Q[:,7],))
    M = csr_matrix((vals,(I,I)),shape=(V.shape[0],V.shape[0]))
    return M