import numpy as np
from gpytoolbox.squared_distance import squared_distance
from gpytoolbox.edge_indices import edge_indices

def unsigned_distance(Q,V,F=None,use_cpp=True,aabb=None):
    """Unsigned distances from a set of points in space.

    General-purpose function which computes the unsigned distance from a set of points to a mesh (in 3D) or polyline (in 2D). In 3D, this uses an AABB tree for efficient computation.

    Parameters
    ----------
    Q : (p,dim) numpy double array
        Matrix of query point positions
    V : (v,dim) numpy double array
        Matrix of mesh/polyline/pointcloud coordinates
    F : (f,s) numpy int array (optional, default None)
        Matrix of mesh/polyline/pointcloud indices into V. If None, input is assumed to be an ordered *closed* polyline in 2D.
    use_cpp : bool, optional (default False)
        If True, uses a C++ implementation to compute the squared distances. This is much faster but requires compilation of the C++ code.
    aabb : gpytoolbox.AABBTree, optional (default None)
        Precomputed AABB tree built via `gpytoolbox.AABBTree(V, F)`. Only used
        when `use_cpp=True`. Reuses the tree across calls instead of rebuilding
        it, avoiding the O(n) construction cost on every query.

    Returns
    -------
    unsigned_distances : (p,) numpy double array
        Vector of minimum unsigned distances
    indices : (p,) numpy int array
        Indices into F (or V, if F is None) of closest elements to each query point
    lmbs : (p,s) numpy double array
        Barycentric coordinates into the closest element of each closest mesh point to each query point

    See Also
    --------
    squared_distance, AABBTree
    """
    # Step 1: Get squared distances
    dim = V.shape[1]
    if F is None:
        # Assume polyline
        assert dim==2
        F = edge_indices(V.shape[0],closed=True)
    sqrD, I, lmbd =  squared_distance(Q,V,F,use_cpp=use_cpp,use_aabb=True,aabb=aabb)

    # Step 2: Compute unsigned distance
    dist = np.sqrt(sqrD)

    return dist, I, lmbd