import numpy as np
from numpy.core.function_base import linspace
from scipy.sparse import csr_matrix


def fd_interpolate(P,gs,h,corner=None):
    """Bi/Trilinear interpolation matrix

    Given a regular finite-difference grid described by the number of nodes on each side, the grid spacing, and the location of the bottom-left-front-most corner node, and a list of points, construct a sparse matrix of bilinear interpolation weights so that P = W @ x

    Parameters
    ----------
    P : numpy double array 
        Matrix of interpolated point coordinates
    gs : numpy int array
        Grid size [nx,ny(,nz)]
    h : numpy double array
        Spacing between grid points [hx,hy(,hz)]
    corner: numpy double array (optional, default None)
        Location of the bottom-left-front-most corner node

    Returns
    -------
    W  : scipy sparse.csr_matrix
        Sparse matrix such that if x are the grid nodes, P = W @ x

    See Also
    --------
    regular_square_mesh.

    Notes
    -----
    The ordering in the output is consistent with the mesh built in regular_square_mesh

    Examples
    --------
    TO-DO
    """

    dim = P.shape[1]

    if corner is None:
        corner = np.zeros(dim)

    indeces = np.floor( (P - np.tile(corner,(P.shape[0],1)))/np.tile(h,(P.shape[0],1)) ).astype(int)
    I = linspace(0,P.shape[0]-1,P.shape[0],dtype=int)
    if dim==2:
        indeces_vectorized = indeces[:,0] + gs[0]*indeces[:,1]
        I = np.concatenate((I,I,I,I))
        J = np.concatenate(( indeces_vectorized,indeces_vectorized+gs[0],indeces_vectorized+1,indeces_vectorized+1+gs[0] ))
    else:
        indeces_vectorized = (indeces[:,1] + gs[1]*indeces[:,2])*gs[0] + indeces[:,0]
        I = np.concatenate((I,I,I,I,I,I,I,I))
        J = np.concatenate(( indeces_vectorized,indeces_vectorized+gs[0],indeces_vectorized+1,indeces_vectorized+1+gs[0], indeces_vectorized+(gs[1]*gs[0]),indeces_vectorized+gs[0]+(gs[1]*gs[0]),indeces_vectorized+1+(gs[1]*gs[0]),indeces_vectorized+1+gs[0]+(gs[1]*gs[0]) ))
    
    # Position in the bottom left corner
    vij = np.tile(corner,(P.shape[0],1)) + indeces*np.tile(h,(P.shape[0],1))
    # Normalized position inside cell
    vij = (P - vij)/np.tile(h,(P.shape[0],1))
    # Coefficients wrt to each corner of cell
    if dim==2:
        coeff_00 = (1-vij[:,1])*(1-vij[:,0]) # bottom left
        coeff_01 = (1-vij[:,1])*vij[:,0] # bottom right
        coeff_10 = vij[:,1]*(1-vij[:,0]) # top left
        coeff_11 = vij[:,1]*vij[:,0] # top right
        # concatenate (watch that order is consistent with J!!)
        vals = np.concatenate((coeff_00,coeff_10,coeff_01,coeff_11))
        mat_dim = gs[0]*gs[1]
    else:
        coeff_000 = (1-vij[:,1])*(1-vij[:,0])*(1-vij[:,2]) # bottom left
        coeff_010 = (1-vij[:,1])*vij[:,0]*(1-vij[:,2]) # bottom right
        coeff_100 = vij[:,1]*(1-vij[:,0])*(1-vij[:,2]) # top left
        coeff_110 = vij[:,1]*vij[:,0]*(1-vij[:,2]) # top right
        coeff_001 = (1-vij[:,1])*(1-vij[:,0])*vij[:,2] # bottom left
        coeff_011 = (1-vij[:,1])*vij[:,0]*vij[:,2] # bottom right
        coeff_101 = vij[:,1]*(1-vij[:,0])*vij[:,2] # top left
        coeff_111 = vij[:,1]*vij[:,0]*vij[:,2] # top right
        # concatenate (watch that order is consistent with J!!)
        vals = np.concatenate((coeff_000,coeff_100,coeff_010,coeff_110,coeff_001,coeff_101,coeff_011,coeff_111))
        mat_dim = gs[0]*gs[1]*gs[2]
    # Build scipy matrix
    # to-do: maybe add warning if this happens?
    I = I[(J>=0)*(J<mat_dim)]
    vals = vals[(J>=0)*(J<mat_dim)]
    J = J[(J>=0)*(J<mat_dim)]
    W = csr_matrix((vals,(I,J)),shape=(P.shape[0],mat_dim))
    return W
