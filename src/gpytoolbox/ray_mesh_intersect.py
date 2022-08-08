import numpy as np

def ray_mesh_intersect(cam_pos,cam_dir,V,F):
    """Shoot a ray from a position and see where it crashes into a given mesh

    Uses a bounding volume hierarchy to efficiently compute intersections of many different rays with a given mesh.

    Parameters
    ----------
    cam_pos : numpy double array
        Matrix of camera positions
    cam_dir : numpy double array
        Matrix of camera directions
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    ts : list of doubles
        The i-th entry of this list is the time it takes a ray starting at the i-th camera position with the i-th camera direction to hit the surface (inf if no hit is detected)
    ids : list of ints
        The i-th entry is the index into F of the mesh element that the i-th ray hits (-1 if no hit is detected)
    lambdas : numpy double array
        The i-th row contains the barycentric coordinates of where in the triangle the ray hit (all zeros is no hit is detected)

    Examples
    --------
    TODO
    """

    try:
        from gpytoolbox_bindings import _ray_mesh_intersect_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ decimate binding.")

    ts, ids, lambdas = _ray_mesh_intersect_cpp_impl(cam_pos.astype(np.float64),cam_dir.astype(np.float64),V.astype(np.float64),F.astype(np.int32))

    return ts, ids, lambdas
