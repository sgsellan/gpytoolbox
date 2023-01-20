import numpy as np
from gpytoolbox.doublearea import doublearea

def barycentric_coordinates(p,v1,v2,v3):
    """Per-vertex weights of point inside triangle

    Computes the barycentric coordinates of a point inside a triangle in two or three dimensions. If 3D, computes the coordinates of the point's *projection* to the triangle's plane.

    Parameters
    ---------
    p : (n,dim) numpy double array
        Query points coordinates
    v1 : (n,dim) numpy double array
        First triangle vertex coordinates
    v2 : (n,dim) numpy double array
        Second triangle vertex coordinates
    v3 : (n,dim) numpy double array
        Third triangle vertex coordinates

    Returns
    -------
    b : (n,3) numpy double array
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
    # If p is a single point, reshape to (1,dim)
    if (p.ndim == 1):
        p = p[None,:]
    # If v1 is a single point, reshape to (1,dim)
    if (v1.ndim == 1):
        v1 = v1[None,:]
    # If v2 is a single point, reshape to (1,dim)
    if (v2.ndim == 1):
        v2 = v2[None,:]
    # If v3 is a single point, reshape to (1,dim)
    if (v3.ndim == 1):
        v3 = v3[None,:]
    

    # p = np.ravel(p)
    # v1 = np.ravel(v1)
    # v2 = np.ravel(v2)
    # v3 = np.ravel(v3)
    dim = p.shape[1]
    n = p.shape[0]
    if dim==2:
        tri1_verts = np.vstack((
            p,
            v2,
            v3
        ))
        tri2_verts = np.vstack((
            v1,
            p,
            v3
        ))
        tri3_verts = np.vstack((
            v1,
            v2,
            p
        ))
        tri_verts = np.vstack((
            v1,
            v2,
            v3
        ))
        tri_inds = np.zeros((n,3),dtype=int)
        tri_inds[:,0] = np.arange(n)
        tri_inds[:,1] = np.arange(n)+n
        tri_inds[:,2] = np.arange(n)+2*n
        a1 = doublearea(tri1_verts,tri_inds,signed=True)
        a2 = doublearea(tri2_verts,tri_inds,signed=True)
        a3 = doublearea(tri3_verts,tri_inds,signed=True)
        a = doublearea(tri_verts,tri_inds,signed=True)
        b = np.zeros((n,3))
        b[:,0] = a1/a
        b[:,1] = a2/a
        b[:,2] = a3/a

        # a1 = np.squeeze(doublearea(tri1_verts,np.array([[0,1,2]]),signed=True))
        # a2 = np.squeeze(doublearea(tri2_verts,np.array([[0,1,2]]),signed=True))
        # a3 = np.squeeze(doublearea(tri3_verts,np.array([[0,1,2]]),signed=True))
        # a = np.squeeze(doublearea(tri_verts,np.array([[0,1,2]]),signed=True))
        # b = np.array([a1/a, a2/a, a3/a])
    elif dim==3:
        u = v2-v1
        v = v3-v1
        N = np.cross(u,v,axis=1)
        w = p-v1
        n2 = np.sum(N**2.0,axis=1)
        gamma = np.sum(np.multiply(np.cross(u,w,axis=1),N),axis=1)/n2
        beta = np.sum(np.multiply(np.cross(w,v,axis=1),N),axis=1)/n2
        alpha = 1 - beta - gamma
        b = np.zeros((n,3))
        b[:,0] = alpha
        b[:,1] = beta
        b[:,2] = gamma
        # b = np.array([alpha,beta,gamma])
    return b