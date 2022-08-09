import numpy as np

def mesh_boolean(V1,F1,V2,F2,boolean_type='union'):
    """Finds intersection, union or subtraction of two triangle meshes.

    Given two triangle meshes dA and dB, uses exact predicates to compute the intersection, union or subtraction of the two solids A and B, and output its surface dC

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
    boolean_type : str, optional (default 'union')
        Operation: one of 'union' (default), 'intersection', 'difference'

    Returns
    -------
    V3 : numpy double array
        Matrix of vertex coordinates of the first mesh
    F3 : numpy int array
        Matrix of triangle indices of the first mesh
    

    See Also
    --------
    do_meshes_intersect.

    Notes
    -----
    'minus', 'difference' and 'subtraction' are alias or one another

    Examples
    --------
    TO-DO
    """

    try:
        from gpytoolbox_bindings_copyleft import _mesh_union_cpp_impl
        from gpytoolbox_bindings_copyleft import _mesh_intersection_cpp_impl
        from gpytoolbox_bindings_copyleft import _mesh_difference_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    dictionary ={
    'union' : 0,
    'intersection' : 1,
    'difference' : 2,
    'minus' : 2
    }
    btype = dictionary.get(boolean_type,-1)
    if btype==0:
        v, f = _mesh_union_cpp_impl(V1,F1.astype(np.int32),V2,F2.astype(np.int32))
    elif btype==1:
        v, f = _mesh_intersection_cpp_impl(V1,F1.astype(np.int32),V2,F2.astype(np.int32))
    elif btype==2:
        v, f = _mesh_difference_cpp_impl(V1,F1.astype(np.int32),V2,F2.astype(np.int32))

    
    return v,f
