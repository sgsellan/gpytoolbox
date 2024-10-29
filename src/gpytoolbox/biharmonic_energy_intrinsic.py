import numpy as np
import scipy as sp
from .cotangent_laplacian_intrinsic import cotangent_laplacian_intrinsic
from .massmatrix_intrinsic import massmatrix_intrinsic

def biharmonic_energy_intrinsic(l_sq,F,
    n=None,
    bc='mixedfem_zero_neumann'):
    """Constructs the biharmonic energy matrix Q such that for a per-vertex function u, the discrete biharmonic energy is u'Qu, using only intrinsic information from the mesh.

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh
    n : int, optional (default None)
        number of vertices in the mesh.
        If absent, will try to infer from F.
    bc : string, optional (default 'mixedfem_zero_neumann')
         Which type of discretization and boundary condition to use.
         Options are: {'mixedfem_zero_neumann', 'hessian', 'curved_hessian'}
         - 'mixedfem_zero_neumann' implements the mixed finite element
         discretization from Jacobson et al. 2010.
         "Mixed Finite Elements for Variational Surface Modeling".
         - 'curved_hessian' implements the Hessian energy from Stein et al.
         2020 "A Smoothness Energy without Boundary Distortion for Curved Surfaces"
         via libigl C++ binding.

    Returns
    -------
    Q : (n,n) scipy csr_matrix
        biharmonic energy matrix

    Examples
    --------
    ```python
    # Mesh in V,F
    import gpytoolbox as gpy
    l_sq = gpy.halfedge_lengths_squared(V,F)
    Q = biharmonic_energy_intrinsic(l_sq,F)
    ```
    
    """

    assert F.shape[1] == 3, "Only works on triangle meshes."
    assert l_sq.shape == F.shape
    assert np.all(l_sq>=0)

    if n==None:
        n = np.max(F)+1

    if bc=='mixedfem_zero_neumann':
        Q = _mixedfem_neumann_laplacian_energy(l_sq, F, n)
    elif bc=='curved_hessian':
        Q = _curved_hessian_energy(l_sq, F, n)
    else:
        assert False, "Invalid bc"

    return Q


def _mixedfem_neumann_laplacian_energy(l_sq, F, n):
    # Q = L' * M^{-1} * L
    L = cotangent_laplacian_intrinsic(l_sq, F, n=n)
    M = massmatrix_intrinsic(l_sq, F, n=n, type='voronoi')
    M_inv = M.power(-1) #Sparse matrix inverts componentwise
    Q = L.transpose() * M_inv * L

    return Q


try:
    # Import C++ bindings for curved Hessian
    from gpytoolbox_bindings import _curved_hessian_intrinsic_cpp_impl
    _CPP_CURVED_HESSIAN_INTRINSIC_AVAILABLE = True
except Exception as e:
    _CPP_CURVED_HESSIAN_INTRINSIC_AVAILABLE = False

def _curved_hessian_energy(l_sq, F, n):
    assert _CPP_CURVED_HESSIAN_INTRINSIC_AVAILABLE, \
    "C++ bindings for curved Hessian not available."

    Q = _curved_hessian_intrinsic_cpp_impl(l_sq.astype(np.float64),
        F.astype(np.int32))
    if Q.shape[0] != n:
        Q = sp.sparse.block_diag([
            Q, sp.sparse.csr_matrix((n-Q.shape[0], n-Q.shape[0]))],
            format='csr')
    else:
        Q = sp.sparse.csr_matrix(Q)

    return Q

