import numpy as np
from gpytoolbox.per_face_normals import per_face_normals
from gpytoolbox.triangle_triangle_adjacency import triangle_triangle_adjacency

def dihedral_angles(V, F):
    """Angle between adjacent faces across each (half-)edge of a triangle mesh.

    For every halfedge of every face, computes the (unsigned) angle between the normal of
    that face and the normal of the face on the other side of the edge (the
    "crease" or "turning" angle). Importantly, this uses the following convention:

    - `0` when the two faces are coplanar (a flat surface),
    - `pi/2` when they meet at a right angle (e.g. an edge of a cube),
    - approaching `pi` when they fold back onto each other.

    Parameters
    ----------
    V : (n,3) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    theta : (m,3) numpy double array
        Per-halfedge dihedral angles in radians, following the halfedge
        ordering convention (entry `(f,i)` is the angle across the halfedge of
        face `f` opposite its vertex `i`, i.e. the edge
        `(F[f,(i+1)%3], F[f,(i+2)%3])`). Boundary halfedges (no adjacent face)
        are assigned `nan`. Each interior edge appears in two entries with the
        same value.

    See Also
    --------
    per_face_normals, triangle_triangle_adjacency.

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, dihedral_angles
    v,f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    theta = dihedral_angles(v,f)
    ```
    """

    assert F.shape[1] == 3, "dihedral_angles only supports triangle meshes"
    assert V.shape[1] == 3, "dihedral_angles only supports 3D meshes"

    m = F.shape[0]
    N = per_face_normals(V, F, unit_norm=True)
    TT, _ = triangle_triangle_adjacency(F)

    theta = np.full((m, 3), np.nan)
    for i in range(3):
        nb = TT[:, i]
        valid = nb >= 0
        if not np.any(valid):
            continue
        n1 = N[valid, :]
        n2 = N[nb[valid], :]
        dotp = np.sum(n1 * n2, axis=1)
        crossn = np.linalg.norm(np.cross(n1, n2, axis=1), axis=1)
        # arctan2(|n1 x n2|, n1 . n2) is a numerically stable angle in [0, pi].
        theta[valid, i] = np.arctan2(crossn, dotp)

    return theta
