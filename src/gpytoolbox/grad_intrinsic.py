import numpy as np
import scipy as sp

from .grad import grad

# Lifted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/grad.m

def grad_intrinsic(l_sq,F,
    n=None,):
    """Intrinsic finite element gradient matrix

    Given a triangle mesh, computes the finite element gradient matrix assuming
    piecewise linear hat function basis.

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh
    n : int, optional (default None)
        number of vertices in the mesh.
        If absent, will try to infer from F.

    Returns
    -------
    G : (2*m,n) scipy sparse.csr_matrix
        Sparse FEM gradient matrix.
        The first m rows ar ethe gradient with respect to the edge (1,2).
        The m rows after that run in a pi/2 rotation counter-clockwise.

    See Also
    --------
    cotangent_laplacian.

    Notes
    -----
    Adapted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/grad_intrinsic.m

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    l = gpy.halfedge_lengths_squared(V,F)
    G = gpy.grad_intrinsic(l_sq,F)
    ```
    """
    

    assert F.shape[1] == 3, "Only works on triangle meshes."
    assert l_sq.shape == F.shape
    assert np.all(l_sq>=0)

    if n==None:
        n = np.max(F)+1
    m = F.shape[0]

    l0 = np.sqrt(l_sq[:,0])[:,None]
    gx = (l_sq[:,1][:,None]-l_sq[:,0][:,None]-l_sq[:,2][:,None]) / (-2.*l0)
    gy = np.sqrt(l_sq[:,2][:,None] - gx**2)
    V2 = np.block([[gx,gy], [np.zeros((m,2))], [l0, np.zeros((m,1))]])
    F2 = np.reshape(np.arange(3*m), (m,3), order='F')
    G2 = grad(V2,F2)
    P = sp.sparse.csr_matrix((np.ones(3*m),(F2.ravel(),F.ravel())),
        shape=(3*m,n))
    G = G2*P

    return G
