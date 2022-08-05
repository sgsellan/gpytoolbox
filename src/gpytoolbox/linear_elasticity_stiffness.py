import numpy as np
from scipy.sparse import csr_matrix, diags, identity, hstack, vstack, block_diag
from gpytoolbox.grad import grad
from gpytoolbox.doublearea import doublearea
from gpytoolbox.massmatrix import massmatrix

def linear_elasticity_stiffness(V,F,K=1.75,mu=0.0115,volumes=None,mass=None):
    """Differential operators needed for linear elasticity calculations

    Returns the linear elastic stiffness and strain matrices for a given shape and material parameters

    Parameters
    ----------
    V : numpy double array
        Matrix of vertex coordinates
    F : numpy int array
        Matrix of triangle indices
    K : double (optional, default 1.75)
        Bulk modulus
    mu : double (optional, default 0.0115)
        Material shear modulus
    volumes : numpy double array (optional, default None)
        Vector with the volumes (in 2D) or areas (in 3D) of each mesh element (if None, will be computed)
    mass : scipy sparse_csr (optional, default None)
        The mesh's sparse mass matrix (if None, will be computed)
    

    Returns
    -------
    K  : scipy sparse.csr_matrix 
        Stiffness matrix
    C : scipy sparse.csr_matrix 
        Constituitive model matrix 
    strain : scipy sparse.csr_matrix 
        Strain matrix
    A : scipy csr_matrix 
        Diagonal element area matrix
    M : scipy sparse.csr_matrix 
        Mass matrix (if input mass is not None, this returns the input)

    See Also
    --------
    linear_elasticity.

    Notes
    -----
    This implementation only works for 2D triangle meshes. Tetrahedral meshes will be supported soon.

    Examples
    --------
    TO-DO
    """

    l = K - (2/3)*mu
    Z = csr_matrix((F.shape[0],V.shape[0]))
    dim = V.shape[1]
    m = F.shape[0]
    I = identity(m)

    if dim==2:
        G = grad(np.concatenate((V, np.zeros((V.shape[0],1))), axis=1),F)
        G = G[0:(2*F.shape[0]),0:(2*F.shape[0])]
        G0 = G[0:F.shape[0],:]
        G1 = G[F.shape[0]:(2*F.shape[0])]
        strain0 = hstack((G0, Z))
        strain1 = hstack((Z, G1))
        strain2 = hstack((G1, G0))
        strain = vstack((strain0,strain1,strain2))
        C = vstack(( hstack(( (l+2*mu)*I,l*I,0*I )),hstack(( l*I,(l+2*mu)*I,0*I )),hstack(( 0*I, 0*I, mu*I )) ) )
        if (volumes is None):
            A = diags(doublearea(V,F)*0.5)
        else:
            A = diags(volumes)
        A = block_diag((A,A,A))
        if (mass is None):
            M = massmatrix(V,F,'voronoi')
        else:
            M = mass
        
        M = block_diag((M,M))
    if dim==3:
        print("DIMENSION 3 NOT SUPPORTED YET")

    Z = csr_matrix((V.shape[0],F.shape[0]))
    D = strain.transpose()
    K = D * A * C * strain
    K = csr_matrix(K)
    
    return K, C, strain, A, M