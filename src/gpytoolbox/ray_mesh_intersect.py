import numpy as np

def ray_mesh_intersect(cam_pos,cam_dir,V,F):
    # TODO 

    from gpytoolbox_bindings import _ray_mesh_intersect_cpp_impl

    ts, ids, lambdas = _ray_mesh_intersect_cpp_impl(cam_pos,cam_dir,V,F.astype(np.int32))

    return ts, ids, lambdas
