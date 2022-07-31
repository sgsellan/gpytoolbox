import numpy as np

def offset_surface(V,F,iso,grid_size):
    # TODO 

    from gpytoolbox_bindings import _offset_surface_cpp_impl

    v, f = _offset_surface_cpp_impl(V,F.astype(np.int32),iso,grid_size)

    return v,f
