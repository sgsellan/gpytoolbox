import numpy as np

def regular_square_mesh(nx, ny=None):
    """Triangle mesh of a square

    Generates a regular triangular mesh of a one by one square by dividing each grid square into two triangles.

    Parameters
    ----------
    nx : int
        number of vertices on the x-axis
    ny : int, optional (default None)
        number of vertices on the y-axis, default nx

    Returns
    -------
    V : numpy double array
        Matrix of triangle mesh vertex coordinates
    F : numpy int array
        Matrix of triangle vertex indices into V

    See Also
    --------
    regular_cube_mesh.

    Notes
    -----
    The ordering of the vertices is increasing by rows and then columns, so [0,0], [h,0], [2*h,0],...,[h,h],[2*h,h],...,[1-h,1],[1,1]

    Examples
    --------
    ```python
    # Generate a 10x10 triangle mesh
    gs = 10
    V, F = gpytoolbox.regular_square_mesh(gs)
    ```
    """

    if ny is None:
        ny = nx

    x, y = np.meshgrid(np.linspace(-1,1,nx),np.linspace(-1,1,ny))
    v = np.stack((x.ravel(), y.ravel()), axis=-1)

    # Indexing algorithm inspired by gptoolbox's create_regular_grid
    # https://github.com/alecjacobson/gptoolbox/blob/master/mesh/create_regular_grid.m
    inds = np.reshape(np.arange(nx*ny), (ny,nx))
    i0 = inds[:-1,:-1].ravel()
    i1 = inds[:-1,1:].ravel()
    i2 = inds[1:,:-1].ravel()
    i3 = inds[1:,1:].ravel()
    f = np.stack((np.concatenate((i0,i0)),
        np.concatenate((i3,i1)),
        np.concatenate((i2,i3))),
        axis=-1)

    return v,f
