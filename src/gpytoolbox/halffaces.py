import numpy as np

def halffaces(T):
    """Given a tet mesh with face tet T, returns all oriented halffaces
    as indices into the vertex array.

    The ordering convention for halfedges is the following:
    [halfface opposite vertex 0,
     halfface opposite vertex 1,
     halfface opposite vertex 2,
     halfface opposite vertex 3]

    Parameters
    ----------
    T : (m,4) numpy int array
        tet index list of a tet mesh

    Returns
    -------
    hf : (m,4,3) numpy int array
        halfface list as per above conventions

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    v,f = gpy.regular_cube_mesh(4)
    # Call to halffaces
    hf = gpy.halffaces(v,f)
    ```
    
    """
    
    assert T.shape[0] > 0
    assert T.shape[1] == 4

    hf = np.block([[[T[:,1][:,None], T[:,2][:,None], T[:,3][:,None]]],
        [[T[:,3][:,None], T[:,2][:,None], T[:,0][:,None]]],
        [[T[:,3][:,None], T[:,0][:,None], T[:,1][:,None]]],
        [[T[:,1][:,None], T[:,0][:,None], T[:,2][:,None]]]]).transpose((1,0,2))
    return hf

