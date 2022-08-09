import numpy as np

def do_meshes_intersect(V1,F1,V2,F2):
    """Finds whether two surface triangle meshes intersect.

    Given two triangle meshes A and B, uses exact predicates to compute whether any pair of triangles triA and triB intersect one another.

    Parameters
    ----------
    V1 : numpy double array
        Matrix of vertex coordinates of the first mesh
    F1 : numpy int array
        Matrix of triangle indices of the first mesh
    V2 : numpy double array
        Matrix of vertex coordinates of the second mesh
    F2 : numpy int array
        Matrix of triangle indices of the second mesh

    Returns
    -------
    b : bool
        True if the meshes have non-trivial intersection, False otherwise
    inters : numpy int array
        One pair of intersecting triangle indices.
    

    See Also
    --------
    mesh_boolean.

    Notes
    -----
    The algorithm stops when it finds one intersection; therefore, the output is not a complete list of all intersecting triangles

    Examples
    --------
    TO-DO
    """
    
    try:
        from gpytoolbox_bindings_copyleft import _do_meshes_intersect_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    inters = list(_do_meshes_intersect_cpp_impl(V1,F1.astype(np.int32),V2,F2.astype(np.int32)))
    b = False
    if len(inters[0])>0:
        b = True
        inters = inters[0]  
    else:
        inters = None
    return b, inters
