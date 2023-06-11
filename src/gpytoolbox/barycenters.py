import numpy as np



def barycenters(V,F):
    """
    List of element barycenters for triangle mesh or polyline

    Parameters
    ----------
    V : (n,3) array
        Vertex coordinates.
    F : (m,3) array
        Face / edge indices.
    
    Returns
    -------
    B : (m,3) array
        Barycenter coordinates, the i-th row is the barycenter of the i-th face.

    Examples
    --------
    ```python
    V = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    F = np.array([[0,1,2],[0,1,3]])
    B = barycenters(V,F)
    ```
    """     

    B = np.zeros((F.shape[0],V.shape[1]))
    for i in range(F.shape[1]):
        B += V[F[:,i],:]
    B /= F.shape[1]
    return B
