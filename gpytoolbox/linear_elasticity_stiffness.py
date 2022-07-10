import numpy as np
from scipy.sparse import csr_matrix, diags, identity, hstack, vstack, block_diag
import igl
from gpytoolbox.grad import grad
from gpytoolbox.doublearea import doublearea
from gpytoolbox.massmatrix import massmatrix

def linear_elasticity_stiffness(V,F,K=1.75,mu=0.0115,volumes=np.array([]),mass=np.array([])):
    # Returns the linear elastic stiffness and strain matrices for a given shape and 
    # material parameters
    #
    # Note: This only works for 2D (d=2) meshes currently
    # TO-DO: Code tet mesh version of this
    #
    # Inputs:
    #       V  #V by d numpy array of vertex positions 
    #       F  #F by d+1 integer numpy array of element indeces into V
    #       Optional:
    #           K bulk modulus
    #           mu material shear modulus
    #           volumes an #F numpy array with the volumes (if d=3) or areas (if d=2) of each mesh element
    #           mass the shape's #V by #V scipy sparse mass matrix (will be computed otherwise)
    # Outputs:
    #       K  #V*d by #V*d scipy csr sparse stiffness matrix
    #       C  #F**(d*(d+1)/2) by #F**(d*(d+1)/2) scipy csr sparse constituitive model matrix 
    #       strain  #F*(d*(d+1)/2) by #V*d scipy csr sparse strain matrix
    #       A  #F*(d*(d+1)/2) by #F*(d*(d+1)/2) scipy csr sparse diagonal element area matrix
    #       M  #V*d by #V*d scipy csr sparse sparse mass matrix

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
        if volumes.shape[0]==0:
            A = diags(doublearea(V,F)*0.5)
        else:
            A = diags(volumes)
        A = block_diag((A,A,A))
        if mass.shape[0]==0:
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