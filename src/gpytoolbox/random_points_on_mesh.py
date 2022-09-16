import numpy as np
from .doublearea import doublearea

def random_points_on_mesh(V,F,
    n,
    rng=np.random.default_rng(),
    distribution='uniform',
    return_indices=False):
    """Samples a mesh V,F according to a given distribution.
    Valid meshes are polylines or triangle meshes.

    Parameters
    ----------
    V : (n_v,d) numpy array
        vertex list of vertex positions
    F : (m,k) numpy int array
        if k==2, interpret input as ordered polyline;
        if k==3 numpy int array, interpred as face index list of a triangle
        mesh
    rng : numpy rng, optional (default: new `np.random.default_rng()`)
        which numpy random number generator to use
    n : int
        how many points to sample
    method : string, optional (default: 'uniform')
        According to which distribution to sample.
        Currently, only uniform is supported.
    return_indices : bool, optional (default: False)
        Whether to return the indices for each element along with the
        barycentric coordinates of the sampled points within each element

    Returns
    -------
    x : (n,d) numpy array
        the n sampled points
    I : (n,) numpy int array
        element indices where sampled points lie (if requested)
    u : (n,k) numpy array
        barycentric coordinates for sampled points in I


    Examples
    --------
    TODO
    
    """

    assert n>=0

    if n==0:
        if return_indices:
            return np.zeros((0,), dtype=V.dtype), \
            np.zeros((0,), dtype=F.dtype), \
            np.zeros((0,), dtype=V.dtype)
        else:
            return np.zeros((0,), dtype=V.dtype)

    k = F.shape[1]
    if k==2:
        if distribution=='uniform':
            I,u = _uniform_sample_polyline(V,F,n,rng,distribution)
        else:
            assert False, "distribution not supported"
        x = u[:,0][:,None]*V[F[I,0],:] + \
        u[:,1][:,None]*V[F[I,1],:]
    elif k==3:
        if distribution=='uniform':
            I,u = _uniform_sample_triangle_mesh(V,F,n,rng,distribution)
        else:
            assert False, "distribution not supported"
        x = u[:,0][:,None]*V[F[I,0],:] + \
        u[:,1][:,None]*V[F[I,1],:] + \
        u[:,2][:,None]*V[F[I,2],:]
    else:
        assert False, "Only polylines and triangles supported"

    if return_indices:
        return x, I, u
    else:
        return x


def _uniform_sample_polyline(V,E,n,rng,distribution):
    l = np.linalg.norm(V[E[:,1],:] - V[E[:,0],:], axis=1)
    w = l / np.sum(l)

    I = rng.choice(w.shape[0], size=(n,), p=w)

    r = rng.uniform(size=(n,))
    u = np.stack([r, 1.-r], axis=-1)

    return I, u


def _uniform_sample_triangle_mesh(V,F,n,rng,distribution):
    # Adapted partially from code by Justin Solomon, and math from
    # https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d

    A = doublearea(V,F)
    w = A / np.sum(A)

    I = rng.choice(w.shape[0], size=(n,), p=w)

    r = rng.uniform(size=(n,2))
    r2 = r[:,0]
    sqrtr = np.sqrt(r[:,1])
    u = np.stack([1.-sqrtr,  sqrtr * (1.-r2),  r2*sqrtr], axis=-1)

    return I, u
