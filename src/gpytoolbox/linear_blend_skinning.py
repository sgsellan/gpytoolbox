import numpy as np

def linear_blend_skinning(V,Ws,Rs,Ts):
    """
    Deform a mesh using linear blend skinning, given a list of weights, rotations and translations.
    
    Parameters
    ----------
    V : (n,3) numpy array
        vertex list of a triangle mesh
    Ws : (n,m) numpy array
        weights for each vertex and each handle
    Rs : (m,3,3) numpy array
        rotations for each handle
    Ts : (m,3) numpy array
        translations for each handle
        
    Returns
    -------
    U : (n,3) numpy array
        deformed vertex list of a triangle mesh

    Examples
    --------
    ```python
    import numpy as np
    import gpytoolbox as gpt
    # Create a mesh
    V = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    F = np.array([[0,1,2],[1,2,3]])
    # Create a set of handles
    Rs = np.array([np.eye(3),np.eye(3)])
    Ts = np.array([[0,0,0],[1,0,0]])
    # Create a set of weights
    Ws = np.array([[1,0],[0,1],[0,1],[1,0]])
    # Deform the mesh
    U = gpt.linear_blend_skinning(V,Ws,Rs,Ts)
    # Check the result
    assert np.allclose(U,np.array([[0,0,0],[2,0,0],[1,1,0],[1,1,0]]))
    ```
    """

    U = np.zeros_like(V,dtype=float)
    # print(Ws.shape[1])
    for i in range(Ws.shape[1]):
        rotations = np.dot(V,Rs[i].T)
        rep_weights = np.repeat(Ws[:,i,None],V.shape[1],axis=1)
        rep_translations = np.repeat(Ts[i][None,:],V.shape[0],axis=0)
        U += rep_weights*(rotations + rep_translations)
        # print(U)
    return U