import numpy as np
from gpytoolbox.doublearea import doublearea

def barycentric_coordinates(p,v1,v2,v3):
    """Per-vertex weights of point inside triangle

    Computes the barycentric coordinates of a point inside a triangle in two or three dimensions. If 3D, computes the coordinates of the point's *projection* to the triangle's plane.

    Parameters
    ---------
    p : (dim,) numpy double array
        Query point coordinates
    v1 : (dim,) numpy double array
        First triangle vertex coordinate
    v2 : (dim,) numpy double array
        Second triangle vertex coordinate
    v3 : (dim,) numpy double array
        Third triangle vertex coordinate

    Returns
    -------
    b : (3,) numpy double array
        Vector of barycentric coordinates

    Examples
    --------
    ```python
    from gpytoolbox import barycentric_coordinates
    # Generate random triangle points
    v1 = np.random.rand(3)
    v2 = np.random.rand(3)
    v3 = np.random.rand(3)
    # Generate random query point using barycentric coordinates
    q = 0.2*v1 + 0.3*v2 + 0.5*v3
    # Find barycentric coordinates
    b = gpytoolbox.barycentric_coordinates(q,v1,v2,v3)
    ```
    """
    p = np.ravel(p)
    v1 = np.ravel(v1)
    v2 = np.ravel(v2)
    v3 = np.ravel(v3)
    dim = p.shape[0]
    if dim==2:
        tri1_verts = np.vstack((
            p[None,:],
            v2[None,:],
            v3[None,:]
        ))
        tri2_verts = np.vstack((
            v1[None,:],
            p[None,:],
            v3[None,:]
        ))
        tri3_verts = np.vstack((
            v1[None,:],
            v2[None,:],
            p[None,:]
        ))
        tri_verts = np.vstack((
            v1[None,:],
            v2[None,:],
            v3[None,:]
        ))
        a1 = np.squeeze(doublearea(tri1_verts,np.array([[0,1,2]]),signed=True))
        a2 = np.squeeze(doublearea(tri2_verts,np.array([[0,1,2]]),signed=True))
        a3 = np.squeeze(doublearea(tri3_verts,np.array([[0,1,2]]),signed=True))
        a = np.squeeze(doublearea(tri_verts,np.array([[0,1,2]]),signed=True))
        b = np.array([a1/a, a2/a, a3/a])
    elif dim==3:
        u = v2-v1
        v = v3-v1
        n = np.cross(u,v)
        w = p-v1
        n2 = np.sum(n**2.0)
        gamma = np.dot(np.cross(u,w),n)/n2
        beta = np.dot(np.cross(w,v),n)/n2
        alpha = 1 - beta - gamma
        b = np.array([alpha,beta,gamma])
    return b