import numpy as np
import scipy as sp
from gpytoolbox.halfedge_edge_map import halfedge_edge_map

def subdivide(V,F,
    method='upsample',
    iters=1,
    return_matrix=False):
    """Builds the (pos. def.) cotangent Laplacian for a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of vertex positions
    F : numpy int array
        if (m,2), interpret input as ordered polyline;
        if (m,3) numpy int array, interpred as face index list of a triangle
        mesh
    method : string, optional (default: 'upsample')
        Which method to use for subdivison.
        Can be 'upsample' {default}, 'loop' (only triangle meshes)
    iters : int, optional (default: 1)
        How many iterations of subdivision to perform.
    return_matrix : bool, optional (default: False)
        Whether to return the matrix for the sparse map S.

    Returns
    -------
    Vu : (n_u,d) numpy array
        vertex list of subdivided polyline / mesh
    Fu : (m_u,2) or (m_u) numpy int array
        face index list of upsampled polyline or triangle mesh
    S : (n_u,n) sparse scipy csr_matrix (if requested)
        sparse matrix such that `Vu == S*V`;
        returned only if `return_matrix == True`.

    Examples
    --------
    TODO
    
    """

    assert iters >= 0

    Vu,Fu = V,F
    if return_matrix:
        S = sp.sparse.eye(_n(V,F), format='csr')

    for i in range(iters):
        Vu,Fu,St = _one_subdivision(Vu,Fu,method,return_matrix)
        if return_matrix:
            S = St*S

    # If there were no iterations at all, we want to return a copy of the
    # original.
    if iters==0:
        Vu = None if V is None else V.copy()
        Fu = F.copy()

    if return_matrix:
        return Vu, Fu, S
    else:
        return Vu, Fu


def _one_subdivision(V,F,method,return_matrix):
    # Dispatcher function for one single subdivision.
    k = F.shape[1]
    if k==2:
        if method=='upsample':
            Vu,Fu,S = _upsample_polyline(V,F,return_matrix)
        else:
            assert False, "Method not supported for this simplex size."
    elif k==3:
        if method=='upsample':
            Vu,Fu,S = _upsample_triangle_mesh(V,F,return_matrix)
        elif method=='loop':
            Vu,Fu,S = _loop_triangle_mesh(V,F,return_matrix)
        else:
            assert False, "Method not supported for this simplex size."

    else:
        assert False, "Simplex dimension not supported."

    return Vu, Fu, S


def _upsample_polyline(V,F,return_matrix):
    # Upsampling of every polyline by inserting a vertex in the middle of each
    # segment.
    assert V.shape[0]>1
    assert F.shape[1]==2
    assert F.shape[0]>0

    n = _n(V,F)
    m = F.shape[0]
    new = np.arange(n,n+m)
    Fu = np.block([
        [F[:,0], new],
        [new,    F[:,1]]
        ]).transpose()

    if V is None:
        Vu = None
    else:
        Vnew = 0.5*(V[F[:,0],:] + V[F[:,1],])
        Vu = np.concatenate((V,Vnew), axis=0)

    if return_matrix:
        old = np.arange(0,n)
        i = np.concatenate((
            old,
            new,
            new
            ))
        j = np.concatenate((
            old,
            F[:,0],
            F[:,1]
            ))
        k = np.concatenate((
            np.full(n,1.),
            np.full(m,0.5),
            np.full(m,0.5)
            ))
        S = sp.sparse.csr_matrix((k,(i,j)), shape=(n+m,n))
    else:
        S = None

    return Vu,Fu,S


def _upsample_triangle_mesh(V,F,return_matrix,
    return_halfedge_edge_map=False):
    # Upsampling of every triangle by inserting a vertex in the middle of each
    # edge.
    assert F.shape[1]==3
    assert F.shape[0]>0

    n = _n(V,F)
    m = F.shape[0]
    he,E,he_to_E,E_to_he = halfedge_edge_map(F, assume_manifold=True)
    e = E.shape[0]

    # A triangle [0,1,2] is replaced with four new triangles:
    # [[0,3,5], [3,1,4], [5,4,2], [5,3,4]]
    # Vertex k+3 is in the middle of the edge from k to k+1
    Fu = np.block([
        [F[:,0],         n+he_to_E[:,2], n+he_to_E[:,1], n+he_to_E[:,1]],
        [n+he_to_E[:,2], F[:,1],         n+he_to_E[:,0], n+he_to_E[:,2]],
        [n+he_to_E[:,1], n+he_to_E[:,0], F[:,2],         n+he_to_E[:,0]]
        ]).transpose()

    if V is None:
        Vu = None
    else:
        assert V.shape[0]>1
        Vnew = 0.5*(V[E[:,0],:] + V[E[:,1],:])
        Vu = np.concatenate((V,Vnew), axis=0)

    if return_matrix:
        old = np.arange(0,n)
        new = np.arange(n,n+e)
        i = np.concatenate((
            old,
            new,
            new
            ))
        j = np.concatenate((
            old,
            E[:,0],
            E[:,1]
            ))
        k = np.concatenate((
            np.full(n,1.),
            np.full(e,0.5),
            np.full(e,0.5)
            ))
        S = sp.sparse.csr_matrix((k,(i,j)), shape=(n+e,n))
    else:
        S = None

    if return_halfedge_edge_map:
        return Vu,Fu,S,he,E,he_to_E,E_to_he
    else:
        return Vu,Fu,S

def _loop_triangle_mesh(V,F,return_matrix):
    # Apply one iteration of Loop subdivision.
    # This, by default, uses the movement rule and β from
    # https://graphics.stanford.edu/~mdfisher/subdivision.html
    # (not Loop's original β)

    n = _n(V,F)
    m = F.shape[0]
    _,Fu,_,he,E,he_to_E,E_to_he = \
    _upsample_triangle_mesh(None,F,False,return_halfedge_edge_map=True)
    e = E.shape[0]

    # Group halfedges
    bE_mask = (E_to_he[:,1,:] == -1).any(axis=-1)
    bdry_he = E_to_he[bE_mask,0,:]
    int_he = np.concatenate((E_to_he[~bE_mask,0,:],E_to_he[~bE_mask,1,:]), axis=0)
    iV_mask = np.full(n, True)
    iV_mask[E[bE_mask,:].ravel()] = False
    int_tail_he = np.stack(np.nonzero(iV_mask[he[:,:,0]]), axis=-1)

    # Compute, for each vertex, the β needed for Loop.
    n_adj = np.bincount(E.ravel(), minlength=n)
    n_adj[n_adj==0] = -1
    β = np.where(n_adj<3, np.NAN, np.where(n_adj==3, 3./16., (3./8.)/n_adj))

    # We always compute the matrix S since we need it to construct Vu
    i = np.concatenate((
        # Add new boundary vertex in the middle of each bdry_he halfedge
        n+he_to_E[bdry_he[:,0], bdry_he[:,1]],
        n+he_to_E[bdry_he[:,0], bdry_he[:,1]],
        # Move old bdry vertex based on 3/8 1/8 1/8 rule
        he[bdry_he[:,0], bdry_he[:,1], 0],
        he[bdry_he[:,0], bdry_he[:,1], 0],
        he[bdry_he[:,0], bdry_he[:,1], 1],
        # Add new interior vertex with 3/8 3/8 1/8 1/8 rule
        n+he_to_E[int_he[:,0], int_he[:,1]],
        n+he_to_E[int_he[:,0], int_he[:,1]],
        # Move old interior vertex with β, 1-βn rule
        he[int_tail_he[:,0], int_tail_he[:,1], 0],
        he[int_tail_he[:,0], int_tail_he[:,1], 0]
        ))
    j = np.concatenate((
        # Add new boundary vertex in the middle of each bdry_he halfedge
        he[bdry_he[:,0], bdry_he[:,1], 0],
        he[bdry_he[:,0], bdry_he[:,1], 1],
        # Move old bdry vertex based on 3/8 1/8 1/8 rule
        he[bdry_he[:,0], bdry_he[:,1], 0],
        he[bdry_he[:,0], bdry_he[:,1], 1],
        he[bdry_he[:,0], bdry_he[:,1], 0],
        # Add new interior vertex with 3/8 3/8 1/8 1/8 rule
        he[int_he[:,0], int_he[:,1], 0],
        F[int_he[:,0], int_he[:,1]],
        # Move old interior vertex with β, 1-βn rule
        he[int_tail_he[:,0], int_tail_he[:,1], 0],
        he[int_tail_he[:,0], int_tail_he[:,1], 1]
        ))
    k = np.concatenate((
        # Add new boundary vertex in the middle of each bdry_he halfedge
        np.full(bdry_he.shape[0], 0.5),
        np.full(bdry_he.shape[0], 0.5),
        # Move old bdry vertex based on 3/4 1/8 1/8 rule
        np.full(bdry_he.shape[0], 0.75),
        np.full(bdry_he.shape[0], 0.125),
        np.full(bdry_he.shape[0], 0.125),
        # Add new interior vertex with 3/8 3/8 1/8 1/8 rule
        np.full(int_he.shape[0], 0.375),
        np.full(int_he.shape[0], 0.125),
        # Move old interior vertex with β, 1-βn rule
        (1./n_adj[he[int_tail_he[:,0], int_tail_he[:,1],0]] - 
            β[he[int_tail_he[:,0] , int_tail_he[:,1],0]]),
        β[he[int_tail_he[:,0], int_tail_he[:,1],0]]
        ))
    S = sp.sparse.csr_matrix((k,(i,j)), shape=(n+e,n))

    if V is None:
        Vu = None
    else:
        Vu = S*V

    if not return_matrix:
        S = None

    return Vu,Fu,S


def _n(V,F):
    if V is None:
        return np.max(F) + 1
    else:
        return V.shape[0]