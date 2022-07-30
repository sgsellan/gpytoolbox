import numpy as np

def mesh_boolean(V1,F1,V2,F2,boolean_type='union'):
    # TODO 

    from gpytoolbox_bindings import _mesh_union_cpp_impl
    from gpytoolbox_bindings import _mesh_intersection_cpp_impl
    from gpytoolbox_bindings import _mesh_difference_cpp_impl

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
