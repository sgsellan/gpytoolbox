import numpy as np
import warnings
from gpytoolbox.boundary_vertices import boundary_vertices
from gpytoolbox.halfedge_lengths import halfedge_lengths
from gpytoolbox.non_manifold_edges import non_manifold_edges


def remesh_botsch(V, F, i=10, h=None, project=True, feature=np.array([], dtype=int)):
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
    ```
    """
    try:
        from gpytoolbox_bindings import _remesh_botsch_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    if h is None:
        h = np.mean(halfedge_lengths(V, F))

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

    v, f = _remesh_botsch_cpp_impl(V, F.astype(np.int32), i, h, feature, project)

    return v, f
