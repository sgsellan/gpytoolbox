import numpy as np


def octree_marching_cubes(sdf, subdivide=None, lo=(-1.0, -1.0, -1.0),
                          hi=(1.0, 1.0, 1.0), max_depth=6, isovalue=0.0,
                          interpolation='linear', binary_search_steps=10):
    """Extracts a level set of a function with marching cubes on an adaptive
    octree.

    The octree is refined top-down wherever `subdivide` says so, down to
    `max_depth`, then balanced so that leaves sharing a face differ by at
    most one level. Marching cubes runs on every leaf, and the cracks between
    leaves of different sizes are closed with the transition cells of the
    Transvoxel algorithm (E. Lengyel, "Voxel-Based Terrain for Real-Time
    Virtual Simulations", PhD thesis, 2010), so the output mesh is
    watertight. Transition cells take a width of a quarter of the larger
    leaf: its vertices near the transition face are moved inward, along the
    surface.

    Parameters
    ----------
    sdf : callable
        The function whose level set is extracted (e.g., a signed distance
        function): takes a (k,3) numpy double array of points and returns
        their (k,) values. Negative values are inside. It is called a few
        times on many points (once for all the octree corners, once per
        binary search step, once for the normals of the moved vertices),
        so it should be vectorized.
    subdivide : callable, optional (default None)
        Which cells to split: takes the cells' minimum corners `cell_lo`,
        (k,3) numpy double array, maximum corners `cell_hi`, (k,3) numpy
        double array, and depths `depth`, (k,) numpy int array (0 for the
        root), and returns a (k,) boolean array, True for the cells to split
        into 8 children. It is called once per depth, with all the cells of
        that depth that are shallower than `max_depth`. Balancing may split
        a few more cells. By default, cells are split where the level set
        may cross them, assuming `sdf` is 1-Lipschitz (as a signed distance
        function is): `|sdf(center) - isovalue|` at most half the cell's
        diagonal.
    lo : (3,) array_like, optional (default (-1,-1,-1))
        Minimum corner of the octree's root cell.
    hi : (3,) array_like, optional (default (1,1,1))
        Maximum corner of the octree's root cell.
    max_depth : int, optional (default 6)
        Maximum depth of the octree (0 to 20). The smallest cells are
        `(hi - lo) / 2**max_depth` wide.
    isovalue : float, optional (default 0.0)
        Level set to extract.
    interpolation : str, optional (default 'linear')
        How the vertex is placed on a cell edge where the function changes
        sign: 'linear' interpolates the endpoint values (no extra
        evaluations); 'binary_search' bisects the edge
        `binary_search_steps` times, evaluating `sdf` once per step for all
        the vertices.
    binary_search_steps : int, optional (default 10)
        Number of binary search steps, if `interpolation` is 'binary_search'.

    Returns
    -------
    V : (n,3) numpy double array
        Matrix of mesh vertex positions
    F : (m,3) numpy int array
        Matrix of triangle indices into V, oriented with normals pointing
        outside (towards values above `isovalue`)

    See Also
    --------
    marching_cubes, dual_contouring_of_signed_distance_data

    Notes
    -----
    Where the surface leaves the root cell, the mesh ends on its faces.

    Examples
    --------
    ```python
    # A sphere of radius 0.6, refined near the surface
    sdf = lambda P: np.linalg.norm(P, axis=1) - 0.6
    V, F = gpytoolbox.octree_marching_cubes(sdf, max_depth=7)

    # A signed distance to a mesh, refined only where the surface is curved:
    # split cells near the surface where the trilinear interpolation of the
    # corners misses the function at the center
    sdf = lambda P: gpytoolbox.signed_distance(P, V0, F0)[0]
    def subdivide(cell_lo, cell_hi, depth):
        center = 0.5 * (cell_lo + cell_hi)
        half_diagonal = 0.5 * np.linalg.norm(cell_hi - cell_lo, axis=1)
        corners = [np.where(np.array([i & 1, i & 2, i & 4]) > 0, cell_hi, cell_lo)
                   for i in range(8)]
        trilinear = np.mean([sdf(c) for c in corners], axis=0)
        s = sdf(center)
        return (depth < 3) | ((np.abs(s) <= half_diagonal) &
                              (np.abs(s - trilinear) > 1e-3))
    V, F = gpytoolbox.octree_marching_cubes(sdf, subdivide, max_depth=8,
        interpolation='binary_search')
    ```
    """
    try:
        from gpytoolbox_bindings import _octree_marching_cubes_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ octree marching cubes binding.")

    if not callable(sdf):
        raise ValueError("sdf must be callable.")
    if subdivide is not None and not callable(subdivide):
        raise ValueError("subdivide must be callable or None.")
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("lo and hi must have 3 coordinates.")
    if not np.all(lo < hi):
        raise ValueError("lo must be smaller than hi in every coordinate.")
    max_depth = int(max_depth)
    if max_depth < 0 or max_depth > 20:
        raise ValueError("max_depth must be between 0 and 20.")
    if interpolation not in ('linear', 'binary_search'):
        raise ValueError("interpolation must be 'linear' or 'binary_search'.")
    binary_search_steps = int(binary_search_steps)
    if binary_search_steps < 0:
        raise ValueError("binary_search_steps must be nonnegative.")
    isovalue = float(isovalue)

    def field(P):
        values = np.asarray(sdf(P), dtype=np.float64).reshape(-1)
        if values.shape[0] != P.shape[0]:
            raise ValueError(f"sdf returned {values.shape[0]} values for {P.shape[0]} points.")
        return values - isovalue

    if subdivide is None:
        def split(cell_lo, cell_hi, depth):
            center = 0.5 * (cell_lo + cell_hi)
            half_diagonal = 0.5 * np.linalg.norm(cell_hi - cell_lo, axis=1)
            return (np.abs(field(center)) <= half_diagonal).astype(np.float64)
    else:
        def split(cell_lo, cell_hi, depth):
            s = np.asarray(subdivide(cell_lo, cell_hi, depth)).reshape(-1)
            if s.shape[0] != cell_lo.shape[0]:
                raise ValueError(f"subdivide returned {s.shape[0]} values for {cell_lo.shape[0]} cells.")
            return s.astype(np.float64)

    V, F = _octree_marching_cubes_cpp_impl(
        sdf=field,
        subdivide=split,
        lo=lo,
        hi=hi,
        max_depth=max_depth,
        binary_search=(interpolation == 'binary_search'),
        binary_search_steps=binary_search_steps,
    )
    return V, F
