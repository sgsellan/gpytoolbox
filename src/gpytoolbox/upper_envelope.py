import numpy as np

def upper_envelope(V,F,D):
    # TODO 

    from gpytoolbox_bindings import _upper_envelope_cpp_impl

    ut, gt, lt = _upper_envelope_cpp_impl(V,F.astype(np.int32),D)

    return ut, gt, lt
