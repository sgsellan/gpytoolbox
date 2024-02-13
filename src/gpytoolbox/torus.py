import numpy as np

def torus(nR,
    nr,
    R=1.,
    r=0.5):
    """Returns a torus mesh.

    Parameters
    ----------
    nR : int
        number of vertices along the large perimeter of the torus (at least 3)
    nr : int
        number of vertices along the small perimeter of the torus (at least 3)
    R : float, optional (default 1.)
        large radius of torus
    r : float, optional (default 0.5)
        small radius of torus

    Returns
    -------
    V : (n,3) numpy array
        vertex positions of the torus
    F : (m,3) numpy array
        face positions of the torus

    Examples
    --------
    ```python
    >>> import gpytoolbox as gpy
    >>> V,F = gpy.torus(4, 3, R=1., r=0.1)
    >>> V
    array([[ 1.10000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-5.50000000e-01,  9.52627944e-01,  0.00000000e+00],
       [-5.50000000e-01, -9.52627944e-01,  0.00000000e+00],
       [ 1.00000000e+00,  0.00000000e+00,  1.00000000e-01],
       [-5.00000000e-01,  8.66025404e-01,  1.00000000e-01],
       [-5.00000000e-01, -8.66025404e-01,  1.00000000e-01],
       [ 9.00000000e-01,  0.00000000e+00,  1.22464680e-17],
       [-4.50000000e-01,  7.79422863e-01,  1.22464680e-17],
       [-4.50000000e-01, -7.79422863e-01,  1.22464680e-17],
       [ 1.00000000e+00,  0.00000000e+00, -1.00000000e-01],
       [-5.00000000e-01,  8.66025404e-01, -1.00000000e-01],
       [-5.00000000e-01, -8.66025404e-01, -1.00000000e-01]])
    >>> F
    array([[ 0,  4,  3],
       [ 1,  5,  4],
       [ 2,  3,  5],
       [ 3,  7,  6],
       [ 4,  8,  7],
       [ 5,  6,  8],
       [ 6, 10,  9],
       [ 7, 11, 10],
       [ 8,  9, 11],
       [ 9,  1,  0],
       [10,  2,  1],
       [11,  0,  2],
       [ 0,  1,  4],
       [ 1,  2,  5],
       [ 2,  0,  3],
       [ 3,  4,  7],
       [ 4,  5,  8],
       [ 5,  3,  6],
       [ 6,  7, 10],
       [ 7,  8, 11],
       [ 8,  6,  9],
       [ 9, 10,  1],
       [10, 11,  2],
       [11,  9,  0]])
    ```
    
    """

    assert nR>=3, "At least 3 vertices along the large perimeter."
    assert nr>=3, "At least 3 vertices along the small perimeter."

    assert R>0.
    assert r>0.

    φ,θ = np.meshgrid(np.linspace(0., 2.*np.pi, nR, endpoint=False),
        np.linspace(0., 2.*np.pi, nr, endpoint=False))
    x = (R + r*np.cos(θ)) * np.cos(φ)
    y = (R + r*np.cos(θ)) * np.sin(φ)
    z = r * np.sin(θ)
    V = np.stack((x.ravel(),y.ravel(),z.ravel()), axis=-1)

    # Indexing algorithm inspired by gptoolbox's create_regular_grid
    # https://github.com/alecjacobson/gptoolbox/blob/master/mesh/create_regular_grid.m
    inds = np.reshape(np.arange(nr*nR), (nr,nR))
    inds = np.concatenate((inds, inds[:,0][:,None]), axis=-1)
    inds = np.concatenate((inds, inds[0,:][None,:]), axis=0)
    i0 = inds[:-1,:-1].ravel()
    i1 = inds[:-1,1:].ravel()
    i2 = inds[1:,:-1].ravel()
    i3 = inds[1:,1:].ravel()
    F = np.stack((np.concatenate((i0,i0)),
        np.concatenate((i3,i1)),
        np.concatenate((i2,i3))),
        axis=-1)

    return V,F

