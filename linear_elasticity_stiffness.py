import numpy as np
from scipy.sparse import csr_matrix, diags, identity, hstack, vstack, block_diag
import igl


def linear_elasticity_stiffness(V,F,K=1.75,mu=0.0115,volumes=np.array([]),mass=np.array([])):
    l = K - (2/3)*mu
    Z = csr_matrix((F.shape[0],V.shape[0]))
    dim = V.shape[1]
    m = F.shape[0]
    I = identity(m)

    if dim==2:
        G = igl.grad(np.concatenate((V, np.zeros((V.shape[0],1))), axis=1),F)
        G = G[0:(2*F.shape[0]),0:(2*F.shape[0])]
        G0 = G[0:F.shape[0],:]
        G1 = G[F.shape[0]:(2*F.shape[0])]
        strain0 = hstack((G0, Z))
        strain1 = hstack((Z, G1))
        strain2 = hstack((G1, G0))
        strain = vstack((strain0,strain1,strain2))
        C = vstack(( hstack(( (l+2*mu)*I,l*I,0*I )),hstack(( l*I,(l+2*mu)*I,0*I )),hstack(( 0*I, 0*I, mu*I )) ) )
        if volumes.shape[0]==0:
            A = diags(igl.doublearea(V,F)*0.5)
        else:
            A = diags(volumes)
        A = block_diag((A,A,A))
        if mass.shape[0]==0:
            M = igl.massmatrix(V,F)
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