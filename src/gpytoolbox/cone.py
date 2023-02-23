import numpy as np

def cone(nx,nz):
    """Returns a cone mesh.

    Parameters
    ----------
    nx : int
         number of vertices along the base of the cone (at least 3)
    nz : int
         number of vertices on the z-axis of the cone (at least 2)

    Returns
    -------
    V : (n,3) numpy array
        vertex positions of the cone
    F : (m,3) numpy array
        face positions of the cone

    Examples
    --------
    ```python
    >>> import gpytoolbox as gpy
    >>> V,F = gpy.cone(3,2)
    >>> V
    array([[ 1.       ,  0.       ,  0.       ],
       [-0.5      ,  0.8660254,  0.       ],
       [-0.5      , -0.8660254,  0.       ],
       [ 0.       ,  0.       ,  1.       ]])
    >>> F
    array([[0, 1, 3],
       [1, 2, 3],
       [2, 0, 3]])
    ```
    
    """

    assert nx>=3, "At least 3 vertices along the equator."
    assert nz>=2, "At least 2 vertices along the z-axis."

    # The last row is just a single vertex
    nz -= 1 

    φ = np.linspace(0., 2.*np.pi, nx, endpoint=False)
    x = np.cos(φ)
    y = np.sin(φ)
    Vs = np.stack((x,y), axis=-1)
    z = np.linspace(0., 1., nz+1)[:-1]
    zrep = np.repeat(z,nx)[:,None]
    V = np.concatenate((np.tile(Vs,(nz,1))*(1.-zrep), zrep), axis=-1)
    V = np.concatenate((V, [[0.,0.,1.]]), axis=0)

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
    F = np.concatenate((F,
        np.stack((
            np.arange(nx*(nz-1), nx*nz),
            np.concatenate((np.arange(nx*(nz-1)+1, nx*nz), [nx*(nz-1)])),
            np.repeat(V.shape[0]-1, nx)
            ), axis=1)
        ), axis=0)

    return V,F

