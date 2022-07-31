import numpy as np

def do_meshes_intersect(V1,F1,V2,F2):
    # TODO 

    from gpytoolbox_bindings import _do_meshes_intersect_cpp_impl

    
    return _do_meshes_intersect_cpp_impl(V1,F1.astype(np.int32),V2,F2.astype(np.int32))
