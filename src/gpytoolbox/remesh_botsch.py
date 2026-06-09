import numpy as np
import warnings
from gpytoolbox.boundary_vertices import boundary_vertices
from gpytoolbox.halfedge_lengths import halfedge_lengths
from gpytoolbox.non_manifold_edges import non_manifold_edges


def remesh_botsch(V, F, i=10, h=None, project=True, feature=np.array([], dtype=int), feature_edges=None, detect_feature_edges=False, feature_dihedral_threshold=45.0):
    """Remesh a triangular mesh to have a desired edge length

    Use the algorithm described by Botsch and Kobbelt's "A Remeshing Approach to Multiresolution Modeling" to remesh a triangular mesh by alternating iterations of subdivision, collapse, edge flips and collapses.

    Parameters
    ----------
    V : numpy double array
        Matrix of triangle mesh vertex coordinates
    F : numpy int array
        Matrix of triangle mesh indices into V
    i : int, optional (default 10)
        Number of algorithm iterations
    h : double, optional (default 0.1)
        Desired edge length (if None, will pick average edge length)
    feature : numpy int array, optional (default np.array([], dtype=int))
        List of indices of feature vertices that should not change (i.e., they will also be in the output). They will be placed at the beginning of the output array in the same order (as long as they were unique).
    feature_edges : numpy int array, optional (default None)
        #FE by 2 matrix of indices into V marking feature edges (e.g. sharp creases). Unlike `feature` vertices, feature edges can be split (the new midpoint stays on the feature and the two halves remain feature edges), are never flipped, and can only be collapsed along the feature (the surviving vertex stays on the crease). Feature-line vertices slide along the crease and are reprojected onto the original feature polyline.
    detect_feature_edges : bool, optional (default False)
        If True, automatically detect sharp feature edges from the mesh's dihedral angle (using `dihedral_angles`) and add them to `feature_edges`. An edge is considered a feature when the angle between its two adjacent faces exceeds `feature_dihedral_threshold`.
    feature_dihedral_threshold : float, optional (default 45.0)
        Dihedral angle threshold *in degrees* used when `detect_feature_edges` is True. Edges whose adjacent faces meet at an angle larger than this are treated as feature edges.
    project : bool, optional (default True)
        Whether to reproject the mesh to the input (otherwise, it will smooth over iterations).

    Returns
    -------
    U : numpy double array
        Matrix of output triangle mesh vertex coordinates
    G : numpy int array
        Matrix of output triangle mesh indices into U


    Notes
    -----
    The ordering of the output can be somewhat confusing. The output vertices are ordered as follows: [feature vertices, boundary vertices, other vertices]. If a vertex is both a feature and boundary one, it is treated as a feature vertex for the purposes of the ordering. For a more in-depth explanation see [PR #45](https://github.com/sgsellan/gpytoolbox/pull/45).

    Examples
    --------
    ```python
    # Read a mesh
    v, f = gpytoolbox.read_mesh("bunny_oded.obj")
    # Do 20 iterations of remeshing with a target length of 0.01
    u, g = gpytoolbox.remesh_botsch(v, f, 20, 0.01, True)
    # Automatically detect feature edges with a dihedral angle threshold of 30 degrees
    u, g = gpytoolbox.remesh_botsch(v, f, 20, 0.01, True, detect_feature_edges=True, feature_dihedral_threshold=30.0)
    ```
    """
    try:
        from gpytoolbox_bindings import _remesh_botsch_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    if h is None:
        h = np.mean(halfedge_lengths(V, F))

    if feature_edges is None:
        feature_edges = np.zeros((0, 2), dtype=np.int32)
    feature_edges = np.asarray(feature_edges, dtype=np.int32).reshape((-1, 2))

    # Optionally auto-detect sharp feature edges from the dihedral angle and add
    # them to the (possibly empty) user-provided feature edges.
    if detect_feature_edges:
        from gpytoolbox.dihedral_angles import dihedral_angles
        theta = dihedral_angles(V, F.astype(np.int32))
        thresh = np.deg2rad(feature_dihedral_threshold)
        # NaN entries (boundary halfedges) compare False, so they are ignored.
        sharp = theta > thresh
        detected = []
        for c in range(3):
            mask = sharp[:, c]
            if np.any(mask):
                a = F[mask, (c+1) % 3]
                b = F[mask, (c+2) % 3]
                detected.append(np.stack([a, b], axis=1))
        if len(detected) > 0:
            feature_edges = np.concatenate([feature_edges] + detected, axis=0).astype(np.int32)
        # De-duplicate unordered feature edges.
        if feature_edges.shape[0] > 0:
            feature_edges = np.unique(np.sort(feature_edges, axis=1), axis=0).astype(np.int32)

    # check that feature is unique
    if feature.shape[0] > 0:
        if np.unique(feature).shape[0] != feature.shape[0]:
            warnings.warn(
                "Feature array is not unique. We will compute its unique entries and use those as an input. "
                "We recommend you do this yourself to avoid this warning.")

    feature = np.concatenate((feature, boundary_vertices(F)), dtype=np.int32)

    # reorder feature nodes to the beginning of the array (contributed by Michael Jäger)
    if feature.shape[0] > 0:
        # feature indices need to be unique (including the boundary_vertices)
        tmp, ind = np.unique(feature, return_index=True)
        # unique feature array while preserving the order [feature, boundary_vertices]
        feature = tmp[np.argsort(ind)]

        # number of vertices
        n_vertices = V.shape[0]
        # 0 ... n_vertices array
        old_order = np.arange(n_vertices, dtype=np.int32)
        # new order
        order = np.concatenate((feature, np.delete(old_order, feature)), dtype=np.int32)
        # generate tmp array for reordering mesh indices
        tmp = np.empty(n_vertices, dtype=np.int32)
        tmp[order] = old_order  # this line will fail if features are not unique

        # reorder vertex coordinates
        V = V[order]
        # reorder faces
        F = tmp[F]
        # remap feature edges to the new vertex ordering
        if feature_edges.shape[0] > 0:
            feature_edges = tmp[feature_edges]
        # features are now 0 to n_features
        feature = old_order[:feature.shape[0]]

    # check that at least one vertex is not a boundary vertex
    if feature.shape[0] == V.shape[0]:
        warnings.warn("All vertices in the input mesh are either manually specified feature vertices or boundary vertices, meaning that this call to remesh_botsch will be a no-op.")

    # if mesh is non-manifold, this will crash, so avoid it and produce python-catchable error instead
    ne = non_manifold_edges(F)
    if len(ne) > 0:
        # return error
        raise ValueError("Input mesh is non-manifold.")

    v, f = _remesh_botsch_cpp_impl(V, F.astype(np.int32), i, h, feature, feature_edges.astype(np.int32), project)

    return v, f
