import numpy as np

def regular_square_mesh(gs):
    """Triangle mesh of a square

    Generates a regular triangylar mesh of a one by one by one cube by dividing each grid square into two triangle

    Parameters
    ----------
    gs : int
        Number of vertices on each side

    Returns
    -------
    V : numpy double array
        Matrix of triangle mesh vertex coordinates
    T : numpy int array
        Matrix of triangle vertex indices into V

    See Also
    --------
    regular_cube_mesh.

    Notes
    -----
    The ordering of the vertices is increasing by rows and then columns, so [0,0], [h,0], [2*h,0],...,[h,h],[2*h,h],...,[1-h,1],[1,1]

    Examples
    --------
    TODO
    """
    x, y = np.meshgrid(np.linspace(-1,1,gs),np.linspace(-1,1,gs))
    v = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

    f = np.zeros((2*(gs-1)*(gs-1),3),dtype=int)

    a = np.linspace(0,gs-2,gs-1,dtype=int)
    for i in range(0,gs-1):
        f[((gs-1)*i):((gs-1)*i + gs-1),0] = gs*i + a
        f[((gs-1)*i):((gs-1)*i + gs-1),1] = gs*i + a + gs + 1
        f[((gs-1)*i):((gs-1)*i + gs-1),2] = gs*i + a + gs
    for i in range(0,gs-1):
        f[((gs-1)*(i+gs-1)):((gs-1)*(i+gs-1) + gs-1),0] = gs*i + a
        f[((gs-1)*(i+gs-1)):((gs-1)*(i+gs-1) + gs-1),1] = gs*i + a + 1
        f[((gs-1)*(i+gs-1)):((gs-1)*(i+gs-1) + gs-1),2] = gs*i + a + gs + 1
    return v,f
