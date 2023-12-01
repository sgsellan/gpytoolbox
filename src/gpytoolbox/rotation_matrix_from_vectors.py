import numpy as np

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns one vector to another.

    Parameters
    ----------
    vec1: (3,) numpy array
    vec2: (3,) numpy array

    Returns
    -------
    R: (3,3) numpy array such that R.dot(vec1) ~= vec2

    Notes
    -----
    Taken from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    
    The vectors do not need to be unit norm, but the output matrix will always be a pure rotation matrix; i.e., it will only align one vector in the direction of the other, without scaling it.

    Examples
    --------
    ```python
    v1 = np.random.randn(3)
    # normalize
    v1 = v1/np.linalg.norm(v1)
    v2 = np.random.randn(3)
    # normalize
    v2 = v2/np.linalg.norm(v2)
    R = rotation_matrix_from_vectors(v1,v2)
    v1_aligned = R.dot(v1)
    print(v1_aligned)
    print(v2)
    ```
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions