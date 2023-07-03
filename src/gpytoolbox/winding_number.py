import numpy as np
from gpytoolbox.fast_winding_number import fast_winding_number

def winding_number(O, V, F):
    """
    Compute the sum of solid angles subtended by the faces of a mesh at a set of points. In 2D, this outputs the exact winding number by summing up solid angles; in 3D, this uses the fast winding number approximation by Barrill et al. "Fast Winding Numbers for Soups and Clouds" (SIGGRAPH 2018).

    Parameters
    ----------
    O : (p,dim) numpy double array
        Matrix of query point positions
    V : (v,dim) numpy double array
        Matrix of mesh/polyline/pointcloud coordinates (in 2D, this is a polyline)
    F : (f,s) numpy int array
        Matrix of mesh/polyline/pointcloud indices into V
    
    Returns
    -------
    W : (p,) numpy double array
        Vector of winding numbers

    See Also
    --------
    signed_distance, squared_distance, fast_winding_number

    Examples
    --------
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj") # Read a mesh
    v = gpytoolbox.normalize_points(v) # Normalize mesh
    # Generate query points
    P = 2*np.random.rand(num_samples,3)-4
    # Compute winding numbers
    W = gpytoolbox.winding_number(P,v,f)
    ```
    """
    dim = V.shape[1]
    if dim == 2:   
        # Compute solid angles     
        VS = V[F[:, 0], :]
        VD = V[F[:, 1], :]

        # 2D vectors from O to VS and VD
        O2VS = np.expand_dims(O[:, :2], axis=1) - np.expand_dims(VS[:, :2], axis=0)
        O2VD = np.expand_dims(O[:, :2], axis=1) - np.expand_dims(VD[:, :2], axis=0)

        S = - np.arctan2(O2VD[:, :, 0] * O2VS[:, :, 1] - O2VD[:, :, 1] * O2VS[:, :, 0], O2VD[:, :, 0] * O2VS[:, :, 0] + O2VD[:, :, 1] * O2VS[:, :, 1])
        W = np.sum(S, axis=1) / (2 * np.pi)
    elif dim == 3:
        # Compute winding number. It would be nice to have the exact winding number using solid angle in 3D too, but for now this is good for most times I want it.
        W = fast_winding_number(O,V,F)
    return W