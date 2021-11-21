import numpy as np
from scipy.sparse import csr_matrix, diags, identity, hstack, vstack, block_diag
import igl
from . linear_elasticity_stiffness import linear_elasticity_stiffness

def linear_elasticity(V,F,U0,dt=0.1,bb=np.empty((0,1),dtype=np.int32),bc = np.empty((0,1), dtype=np.float64)
    ,Ud0=np.array([]),fext=np.array([]),K=1.75,mu=0.0115,volumes=np.array([]),mass=np.array([])):
    
    if Ud0.shape[0]==0:
        Ud0 = 0*V
    if fext.shape[0]==0:
        fext = 0*V
    if bb.shape[0]>0:
        # THIS ASSUMES 2D
        bb = np.concatenate((bb,bb+V.shape[0]))
        bc = np.concatenate((bc[:,0],bc[:,1]))
    
    K, C, strain, A, M = linear_elasticity_stiffness(V,F,K=K,volumes=volumes,mass=mass,mu=mu)

    
    A = M + (dt**2)*K
    B = M*((dt**2)*np.reshape(fext,(-1, 1),order='F') + np.reshape(U0,(-1, 1),order='F') + dt*np.reshape(Ud0,(-1, 1),order='F'))

    # We don't have linear equality constraints, but we need to define them to mqwf
    Aeq = csr_matrix((0, 0), dtype=np.float64)
    Beq = np.array([], dtype=np.float64)

    U = igl.min_quad_with_fixed(A,-2*np.squeeze(B),bb,bc,Aeq,Beq,True)
    # https://en.m.wikipedia.org/wiki/Von_Mises_yield_criterion
    face_stress_vec = C*strain*U[1]
    if V.shape[1]==2:
        sigma_11 = face_stress_vec[0:F.shape[0]]
        sigma_22 = face_stress_vec[F.shape[0]:(2*F.shape[0])]
        sigma_12 = face_stress_vec[(2*F.shape[0]):(3*F.shape[0])]
        sigma_v = np.sqrt(0.5*(sigma_11*sigma_11 - sigma_11*sigma_22 + sigma_22*sigma_22 + 3*sigma_12*sigma_12))
    elif V.shape[1]==3:
        print("TET MESHES UNSUPPORTED")
    return U[1], sigma_v