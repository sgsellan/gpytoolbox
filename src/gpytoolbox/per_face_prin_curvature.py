import numpy as np

def per_face_prin_curvature(V,F):
    """
    """
    try:
        from gpytoolbox_bindings import _per_face_prin_curvature_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ ray_mesh_intersect binding.")

    PD1, PD2, PC1, PC2 = _per_face_prin_curvature_cpp_impl(V.astype(np.float64),F.astype(np.int32))

    return PD1, PD2, PC1, PC2
