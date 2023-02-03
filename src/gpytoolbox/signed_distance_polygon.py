import numpy as np

def signed_distance_polygon(P,V):
    """Signed distance from a set of points to a given polygon
    
    Parameters
    ----------
    P : numpy double array
        Matrix of query point coordinates
    V : numpy double array
        Matrix of polyline vertex positions in order

    Returns
    -------
    S : numpy double array  
        Vector of signed distances from each query point to the polyline

    Notes
    -----
    This is lifted from https://www.shadertoy.com/view/wdBXRW

    Examples
    --------
    ```python
    # Build a polyline; for example, a square
    V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])
    sample_points = np.array([  [0.0,0.0],
                                [0.3,0.0],
                                [-1.5,0.5],
                                [1.2,0.0]])
    groundtruth_vals = np.array([-1.0,-0.7,0.5,0.2])
    S = gpytoolbox.signed_distance_polygon(sample_points,V)
    ```
    """
    from warnings import warn
    warn('signed_distance_polygon will be deprecated in gpytoolbox\'s 0.1.0 release in favour of the more general signed_distance, which can be called with the same arguments and output', DeprecationWarning, stacklevel=2)
    n = V.shape[0]
    # vectorized dot product
    d = np.sum( (P - np.tile(V[0,:],(P.shape[0],1)))*(P - np.tile(V[0,:],(P.shape[0],1))) ,axis=1)
    s = 1.0
    for i in range(0,n):
        if i==0:
            j = n-1
        else:
            j = i-1
        e = V[j,:] - V[i,:]
        # Skip if the edge is zero length
        if np.sum(e*e)==0:
            continue
        w = (P - np.tile(V[i,:],(P.shape[0],1)))
        dotwe = np.sum( w*np.tile(e,(P.shape[0],1)) ,axis=1)
        dotee = np.tile((e[0]**2) + (e[1]**2),(P.shape[0]))
        b = w - np.tile(e,(P.shape[0],1))*np.tile(np.array([np.clip( dotwe/dotee ,0.0,1.0)]).T,(1,2))
        d = np.minimum(d,np.sum( b*b ,axis=1))    
        cond = np.array([ P[:,1]>=V[i,1], P[:,1]<=V[j,1], (e[0]*w[:,1])>(e[1]*w[:,0]) ])
        all_true = cond.all(axis=0)
        all_false = (~cond).all(axis=0)
        change_sign = all_true + all_false
        s = ((-np.ones((P.shape[0])))**(change_sign))*s
    return s*np.sqrt(d)