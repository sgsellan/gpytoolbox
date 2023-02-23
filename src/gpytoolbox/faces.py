import numpy as np
from .halffaces import halffaces

def faces(T,
    return_boundary_indices=False,
    return_interior_indices=False,
    return_nonmanifold_indices=False):
    """Given a tet mesh with tet indices T, returns all unique unoriented
    faces as indices into the vertex array.

    Parameters
    ----------
    T : (m,4) numpy int array
        tet index list of a tet mesh
    return_boundary_indices : bool, optional (default False)
        whether to return a list of indices into F denoting the boundary faces
    return_interior_indices : bool, optional (default False)
        whether to return a list of indices into F denoting the interior faces
    return_nonmanifold_indices : bool, optional (default False)
        whether to return a list of indices into E denoting the nonmanifold faces

    Returns
    -------
    F : (nf,3) numpy int array
        indices of edges into the vertex array
    boundary_indices : (b,) numpy int array
        list of indices into F of boundary faces, returned only if requested
    interior_indices : (i,) numpy int array
        list of indices into F of interior faces, returned only if requested
    nonmanifold_indices : (nm,) numpy int array
        list of indices into F of nonmanifold faces, returned only if requested

    Notes
    -----
    Boundary faces come first in F, followed by all other faces. Boundary faces are guaranteed to be oriented like in T, while interior faces are oriented such that indices are sorted.

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    v,t = gpy.regular_cube_mesh(4)
    f = gpy.faces(t)
    ```
    
    """

    assert T.shape[0] > 0
    assert T.shape[1] == 4

    #Sort halffaces. Remove duplicates.
    hf = halffaces(T)
    flat_hf = np.concatenate([hf[:,0,:],hf[:,1,:],hf[:,2,:],hf[:,3,:]], axis=0)
    sorted_hf = np.sort(flat_hf, axis=1)
    unique_hf, unique_indices, unique_count = np.unique(sorted_hf, axis=0,
        return_index=True, return_counts=True)

    #Construct face arrays by preserving the unique indices of boundary and
    # picking sorted orientation for interior faces.
    #Boundary faces have only one halfface.
    bdry_faces = flat_hf[unique_indices[unique_count==1],:]
    #Interior faces have two ore more halffaces.
    interior_faces = unique_hf[unique_count>1,:]
    F = np.concatenate([bdry_faces,interior_faces], axis=0)
    assert F.shape == unique_hf.shape

    if return_boundary_indices or return_interior_indices or return_nonmanifold_indices:
        retval = [F]
        if return_boundary_indices:
            retval.append(np.arange(0, bdry_faces.shape[0]))
        if return_interior_indices:
            retval.append(np.where(unique_count>1)[0])
        if return_nonmanifold_indices:
            #Nonmanifold faces have three or more halffaces.
            retval.append(np.where(unique_count>2)[0])
        return retval
    else:
        return F
