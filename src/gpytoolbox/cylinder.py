import numpy as np

def cylinder(nx,nz):
    """Returns a cylinder mesh.

    Parameters
    ----------
    nx : int
         number of vertices along the equator of the cylinder (at least 3)
    nz : int
         number of vertices on the z-axis of the cylinder (at least 2)

    Returns
    -------
    V : (n,3) numpy array
        vertex positions of the cylinder
    F : (m,3) numpy array
        face positions of the cylinder

    Examples
    --------
    ```python
    >>> import gpytoolbox as gpy
    >>> V,F = gpy.cylinder(3,2)
    >>> V
    array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [-1.0000000e+00,  1.2246468e-16,  0.0000000e+00],
       [ 1.0000000e+00, -2.4492936e-16,  0.0000000e+00],
       [ 1.0000000e+00,  0.0000000e+00,  1.0000000e+00],
       [-1.0000000e+00,  1.2246468e-16,  1.0000000e+00],
       [ 1.0000000e+00, -2.4492936e-16,  1.0000000e+00]])
    >>> F
    array([[0, 4, 3],
       [1, 5, 4],
       [2, 3, 5],
       [0, 1, 4],
       [1, 2, 5],
       [2, 0, 3]])
    ```
    
    """

    assert nx>=3, "At least 3 vertices along the equator."
    assert nz>=2, "At least 2 vertices along the z-axis."

    φ = np.linspace(0., 2.*np.pi, nx, endpoint=False)
    x = np.cos(φ)
    y = np.sin(φ)
    Vs = np.stack((x,y), axis=-1)
    z = np.linspace(0., 1., nz)
    V = np.concatenate((np.tile(Vs,(nz,1)), np.repeat(z,nx)[:,None]), axis=-1)

    # Indexing algorithm inspired by gptoolbox's create_regular_grid
    # https://github.com/alecjacobson/gptoolbox/blob/master/mesh/create_regular_grid.m
    inds = np.reshape(np.arange(nx*nz), (nz,nx))
    inds = np.concatenate((inds, inds[:,0][:,None]), axis=-1)
    i0 = inds[:-1,:-1].ravel()
    i1 = inds[:-1,1:].ravel()
    i2 = inds[1:,:-1].ravel()
    i3 = inds[1:,1:].ravel()
    F = np.stack((np.concatenate((i0,i0)),
        np.concatenate((i3,i1)),
        np.concatenate((i2,i3))),
        axis=-1)

    return V,F

