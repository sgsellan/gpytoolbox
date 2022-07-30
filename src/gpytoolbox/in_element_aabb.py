import numpy as np

def in_element_aabb(queries,V,F):
    # TODO 

    from gpytoolbox_bindings import _in_element_aabb_cpp_impl

    i = _in_element_aabb_cpp_impl(queries,V,F.astype(np.int32))

    return i
