import numpy as np

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns one vector to another.

    Parameters
    ----------
    vec1: (2,) or (3,) numpy array
    vec2: (2,) or (3,) numpy array

    Returns
    -------
    R: (2,2) or (3,3) numpy array such that R.dot(vec1) ~= vec2

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

    # normalized
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    
    dim = len(a)
    assert dim == len(b)

    # this is dimension agnostic even if it doesn't seem like it
    v = np.cross(a, b).flatten()
    if any(v): #if not all zeros then 
        if dim==2:
            theta = np.arccos(np.dot(a, b))
            # with correct sign
            if np.cross(a, b) < 0:
                theta = -theta
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            return R
        else:
            assert dim == 3
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        # is it a reflection?
        R = np.eye(dim)
        if np.dot(a, b) < 0:
            R = -R
        return R