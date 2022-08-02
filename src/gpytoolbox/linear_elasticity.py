import numpy as np
from .linear_elasticity_stiffness import linear_elasticity_stiffness
from .min_quad_with_fixed import min_quad_with_fixed


def linear_elasticity(V,F,U0,dt=0.1,bb=None,bc = None
    ,Ud0=None,fext=None,K=1.75,mu=0.0115,volumes=None,mass=None):
    """Linear elastic deformation

    Compute the deformation of a 2D solid object according to the usual linear elasticity model.

    Parameters
    ----------
    V : numpy double array
        Matrix of vertex coordinates
    F : numpy int array
        Matrix of triangle indices
    U0 : numpy double array 
        Matrix of previous displacements
    dt : double (optional, default 0.1) 
        Timestep
    bb : numpy int array (optional, default None)
        Fixed vertex indices into V
    bc : numpy double array (optional, default None)
        Fixed vertex *displacements*
    fext : numpy double array (optional, default None)
        Matrix of external forces (for example, gravity or a load)
    Ud0 : numpy double array (optional, default None)
        Matrix of previous velocity
    K : double (optional, default 1.75)
        Bulk modulus
    mu : double (optional, default 0.0115)
        Material shear modulus
    volumes : numpy double array (optional, default None)
        Vector with the volumes (in 3D) or areas (in 2D) of each mesh element (if None, will be computed)
    mass : scipy sparse_csr (optional, default None)
        The mesh's sparse mass matrix (if None, will be computed)

    Returns
    -------
    U : numpy double array
        Matrix of new displacements
    sigma_v : numpy double array
        Vector of per-element Von Mises stresses

    See Also
    --------
    linear_elasticity_stiffness.

    Notes
    -----
    This implementation only works for 2D triangle meshes. Tetrahedral meshes will be supported soon.

    Examples
    --------
    TO-DO
    """

    if (Ud0 is None):
        Ud0 = 0*V
    if (fext is None):
        fext = 0*V
    if (bb is not None):
        # This is the bit that assumes 2D
        bb = np.concatenate((bb,bb+V.shape[0]))
        bc = np.concatenate((bc[:,0],bc[:,1]))
    
    K, C, strain, A, M = linear_elasticity_stiffness(V,F,K=K,volumes=volumes,mass=mass,mu=mu)

    
    A = M + (dt**2)*K
    B = M*((dt**2)*np.reshape(fext,(-1, 1),order='F') + np.reshape(U0,(-1, 1),order='F') + dt*np.reshape(Ud0,(-1, 1),order='F'))

    U = min_quad_with_fixed(A,c=-1.0*np.squeeze(B),k=bb,y=bc)
    #print(U)
    # https://en.m.wikipedia.org/wiki/Von_Mises_yield_criterion
    face_stress_vec = C*strain*U
    if V.shape[1]==2:
        sigma_11 = face_stress_vec[0:F.shape[0]]
        sigma_22 = face_stress_vec[F.shape[0]:(2*F.shape[0])]
        sigma_12 = face_stress_vec[(2*F.shape[0]):(3*F.shape[0])]
        sigma_v = np.sqrt(0.5*(sigma_11*sigma_11 - sigma_11*sigma_22 + sigma_22*sigma_22 + 3*sigma_12*sigma_12))
    elif V.shape[1]==3:
        print("TET MESHES UNSUPPORTED")
    return U, sigma_v