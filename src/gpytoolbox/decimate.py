import numpy as np

def decimate(V,F,num_faces):
    # TODO 

    from gpytoolbox_bindings import _decimate_cpp_impl

    v, f, i, j = _decimate_cpp_impl(V,F.astype(np.int32),num_faces)

    return v,f,i,j
