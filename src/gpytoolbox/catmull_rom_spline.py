import numpy as np

def catmull_rom_spline(T,P):
    """Sample a curve that interpolates the points in P at times T

    Constructs a Catmull-Rom cubic spline that passes through the points in P in order and samples it at times T

    Parameters
    ----------
    T : (t,) numpy double array
        Vector of query times between 0 and 1
    P : (p,3) numpy double array
        Matrix of points the curve is known to pass through

    Returns
    -------
    PT : (t,3) numpy double array
        Matrix of coordinates of the curve sampled at times in T


    Notes
    -----
    The curve is assumed to be open.

    Examples
    --------
    ```python
    from gpytoolbox import catmull_rom_spline
    P = np.array([[0.0,0.0],[1.0,1.0],[-1.0,2.0],[0.0,3.0]])
    T = np.linspace(0,1,100)
    PT = catmull_rom_spline(T,P)
    ```
    """

    num_keyframes = P.shape[0]
    num_queries = T.shape[0]
    dim = P.shape[1]
    # Assume keyframes are equally spaced in time
    time_keyframes = np.linspace(0,1,num_keyframes)
    tau = time_keyframes[1]

    b = np.floor(T*(num_keyframes-1)).astype(np.int32)
    b[T==1] = b[T==1] - 1
    # print(b)
    tt = (T - time_keyframes[b])/tau

    X0 = P[b,:]
    X1 = P[b+1,:]

    tangent_0 = (P[np.minimum(b+1,num_keyframes-1,dtype=int),:] - P[np.maximum(b-1,0,dtype=int),:])/2.0
    tangent_0[b==0] = 2*tangent_0[b==0]
    tangent_1 = (P[np.minimum(b+2,num_keyframes-1,dtype=int),:] - P[np.maximum(b,0,dtype=int),:])/2.0
    tangent_1[b==(num_keyframes-2)] = 2*tangent_1[b==(num_keyframes-2)]

    tt_tiled = np.tile(tt.reshape(-1,1),(1,dim))

    PT = (2*(tt_tiled**3.0) - 3.0*(tt_tiled**2.0) + 1)*X0 + ((tt_tiled**3.0) - 2.0*(tt_tiled**2.0) + tt_tiled)*tangent_0 + (-2.0*(tt_tiled**3.0) + 3.0*(tt_tiled**2.0))*X1 + (tt_tiled**3.0 - tt_tiled**2.0)*tangent_1

    return PT