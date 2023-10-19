import numpy as np
import scipy as sp
from .halfedge_lengths_squared import halfedge_lengths_squared
from .biharmonic_energy_intrinsic import biharmonic_energy_intrinsic
from .doublearea import doublearea
from .boundary_vertices import boundary_vertices
from .massmatrix import massmatrix
from .grad import grad


def biharmonic_energy(V,F,
    bc='mixedfem_zero_neumann'):
    """Constructs the biharmonic energy matrix Q such that for a per-vertex function u, the discrete biharmonic energy is u'Qu.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    bc : string, optional (default 'mixedfem_zero_neumann')
         Which type of discretization and boundary condition to use.
         Options are: {'mixedfem_zero_neumann', 'hessian', 'curved_hessian'}
         - 'mixedfem_zero_neumann' implements the mixed finite element
         discretization from Jacobson et al. 2010.
         "Mixed Finite Elements for Variational Surface Modeling".
         - 'hessian' implements the Hessian energy from Stein et al. 2018.
         "Natural Boundary Conditions for Smoothing in Geometry Processing".
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
    Q = gpy.biharmonic_energy(V,F)
    ```
    """

    if bc=='hessian':
        return _hessian_energy(V, F)
    else:
        l_sq = halfedge_lengths_squared(V,F)
        return biharmonic_energy_intrinsic(l_sq,F,n=V.shape[0],bc=bc)


def _hessian_energy(V, F):
    assert F.shape[1]==3, "Only works on triangle meshes."

    # Q = G' * A * D * Mtilde * D' * A * G
    n = V.shape[0]
    m = F.shape[0]
    dim = V.shape[1]

    b = boundary_vertices(F)
    i = np.setdiff1d(np.arange(n),b)
    ni = len(i)

    a = doublearea(V, F) / 2.
    A = sp.sparse.spdiags([np.tile(a, dim)], 0,
        m=dim*m, n=dim*m, format='csr')

    M_d = massmatrix(V, F, type='voronoi').diagonal()[i]
    M_d_inv = 1. / M_d
    M_tilde_inv = sp.sparse.spdiags([np.tile(M_d_inv, dim**2)], 0,
        m=ni*dim**2, n=ni*dim**2, format='csr')

    G = grad(V,F)
    Gi = G[:,i]
    Dlist = []
    for d in range(dim):
        Dlist.append(Gi[d*m:(d+1)*m, :])
    Dblock = sp.sparse.bmat([Dlist], format='csr')
    D = sp.sparse.block_diag(dim*[Dblock], format='csr')

    Q = G.transpose() * A * D * M_tilde_inv * D.transpose() * A * G

    return Q

