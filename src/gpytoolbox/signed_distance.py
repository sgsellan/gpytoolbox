import numpy as np
from gpytoolbox.squared_distance import squared_distance
from gpytoolbox.fast_winding_number import fast_winding_number
from gpytoolbox.edge_indices import edge_indices

def signed_distance(Q,V,F=None,use_cpp=True):
    """Squared distances from a set of points in space.

    General-purpose function which computes the squared distance from a set of points to a mesh (in 3D) or polyline (in 2D). In 3D, this uses an AABB tree for efficient computation.

    Parameters
    ----------
    Q : (p,dim) numpy double array
        Matrix of query point positions
    V : (v,dim) numpy double array
        Matrix of mesh/polyline/pointcloud coordinates
    F : (f,s) numpy int array (optional, default None)
        Matrix of mesh/polyline/pointcloud indices into V. If None, input is assumed to be an ordered *closed* polyline in 2D.
    use_cpp : bool, optional (default False)
        If True, uses a C++ implementation. This is much faster but requires compilation of the C++ code.
    
    Returns
    -------
    signed_distances : (p,) numpy double array
        Vector of minimum signed distances
    indices : (p,) numpy int array
        Indices into F (or V, if F is None) of closest elements to each query point
    lmbs : (p,s) numpy double array
        Barycentric coordinates into the closest element of each closest mesh point to each query point
    
    See Also
    --------
    squared_distance, winding_number

    Examples
    --------
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj") # Read a mesh
    v = gpytoolbox.normalize_points(v) # Normalize mesh
    # Generate query points
    P = 2*np.random.rand(num_samples,3)-4
    # Compute distances
    signed_distance,ind,b = gpytoolbox.squared_distance(P,v,F=f,use_aabb=True)
    ```
    """
    # Step 1: get squared distances
    dim = V.shape[1]
    if F is None:
        # Assume polyline
        assert dim==2
        F = edge_indices(V.shape[0],closed=True)
    sqrD, I, lmbd = squared_distance(Q,V,F,use_cpp=use_cpp,use_aabb=True)
    
    # Step 2: get the signs
    
    if dim == 2:
        assert(F.shape[1] == 2)
        # Get the sign in linear complexity using the trick in https://www.shadertoy.com/view/wdBXRW , would be nice to have a log n BVH implementation of this sign for 2D
        n = F.shape[0]
        s = 1.0
        for i in range(0,n):
            # if i==0:
            #     j = n-1
            # else:
            #     j = i-1
            # e = V[j,:] - V[i,:]
            vj = V[F[i,0],:]
            vi = V[F[i,1],:]
            e = vj - vi
            # Skip if the edge is zero length
            if np.sum(e*e)==0:
                continue
            w = (Q - np.tile(V[i,:],(Q.shape[0],1)))  
            cond = np.array([ Q[:,1]>=vi[1], Q[:,1]<=vj[1], (e[0]*w[:,1])>(e[1]*w[:,0]) ])
            all_true = cond.all(axis=0)
            all_false = (~cond).all(axis=0)
            change_sign = all_true + all_false
            s = ((-np.ones((Q.shape[0])))**(change_sign))*s
        signed_distance = np.sqrt(sqrD)*s
        # print(s)

    else:
        assert(F.shape[1] == 3)
        assert(dim == 3)
        # Compute winding number
        W = fast_winding_number(Q,V,F)
        W = np.sign(2*W-1)
        # Compute signed distance
        dist = np.sqrt(sqrD)
        signed_distance = W*dist
    
    return signed_distance, I, lmbd

    