import numpy as np
import scipy as sp
import gpytoolbox as gpy
import time
import math
import platform, os, sys

from .point_cloud_to_mesh import point_cloud_to_mesh


def reach_for_the_arcs(U, S,
    rng_seed=3452,
    return_point_cloud=False,
    fine_tune_iters=3,
    batch_size=10000,
    num_rasterization_spheres=0,
    screening_weight=10.,
    rasterization_resolution=None,
    max_points_per_sphere=3,
    n_local_searches=None,
    local_search_iters=20,
    local_search_t=0.01,
    tol=1e-4,
    clamp_value=np.inf,
    force_cpu=False,
    parallel=False,
    verbose=False):
    """Creates a mesh from a signed distance function (SDF) using the
    "Reach for the Arcs" method of S. Sellán,  Y. Ren, C. Batty, and O. Stein [2024].
    This works for polylines in 2D or triangle meshes in 3D.

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    return_point_cloud : bool, optional (default False)
        return the reconstructed point cloud normals or not
    fine_tune_iters : int, optional (default 5)
        how may iterations to do in the fine tuning step
    batch_size : int, optional (default 1000)
        how many points in one batch. Set to 0 to disable batching.
    num_rasterization_spheres : int, optional (default 0)
        how many spheres to use at most in the rasterization step.
        Set to zero to use all spheres.
    screening_weight : float, optional (default 0.1)
        PSR screening weight
    rasterization_resolution : int, optional (default 8*n_sdf**(1/d))
        the resolution of the rasterization grid
    max_points_per_sphere : int, optional (default 3)
        How many points should there be at most per sphere.
        Set it to 1 to not add any additional points to any sphere when
        fine-tuning.
        Has to be at least 1 (this is a theoretical minimum).
        If set to larger than 1, spheres that the reconstructed surface
        intersects will have at most max_points_per_sphere added.
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere in
        the locally make feasible step
    local_search_t : double, optional (default 5e-3)
        how far to move each point in the local search for each point in the
        locally make feasible step
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    clamp_value : float, optional (default np.inf)
        value to which the SDF is clamped for clamped SDF reconstruction
    force_cpu : bool, optional (default False)
        whether to force rasterization onto the CPU
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    V : (n,d) numpy array
        reconstructed vertices
    F : (m,d) numpy array
        reconstructed polyline / triangle mesh indices
    P : (n_p,d) numpy array, if requested
        reconstructed point cloud points
    N : (n_p,d) numpy array, if requested
        reconstructed point cloud normals
    """

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if rasterization_resolution is None:
        rasterization_resolution = 64 * math.ceil(n_sdf**(1./d)/16.)
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    # Resize the SDF points in U and the SDF samples in S so it's in [0,1]^d
    trans = np.min(U, axis=0)
    U = U - trans[None,:]
    scale = np.max(U)
    U /= scale
    S = S/scale
    clamp_value = clamp_value/scale


    if verbose:
        print(f" --- Starting Reach for the Arcs --- ")
        t0_total = time.time()

    if verbose:
        print(f"SDF to point cloud...")
        t0_sdf_to_point_cloud = time.time()

    # Split the SDF into positive and negative spheres?
    separate_inside_outside = True
    if separate_inside_outside:
        neg = S<0
        pos = np.logical_not(neg)
        pos,neg = np.nonzero(pos)[0], np.nonzero(neg)[0]
        U_pos, U_neg = U[pos,:], U[neg,:]
        S_pos, S_neg = S[pos], S[neg]
        if pos.size > 0:
            if verbose:
                print(f"  positive spheres")
            P_pos,N_pos,f_pos = _sdf_to_point_cloud(U_pos, S_pos,
                rng_seed=seed(),
                rasterization_resolution=rasterization_resolution,
                n_local_searches=n_local_searches,
                local_search_iters=local_search_iters,
                batch_size=batch_size,
                num_rasterization_spheres=num_rasterization_spheres,
                force_cpu=force_cpu,
                tol=tol, clamp_value=clamp_value,
                parallel=parallel, verbose=verbose)
        else:
            P_pos,N_pos,f_pos = None,None,None
        if neg.size>0:
            if verbose:
                print(f"  negative spheres")
            P_neg,N_neg,f_neg = _sdf_to_point_cloud(U_neg, S_neg,
                rng_seed=seed(),
                rasterization_resolution=rasterization_resolution,
                n_local_searches=n_local_searches,
                local_search_iters=local_search_iters,
                batch_size=batch_size,
                num_rasterization_spheres=num_rasterization_spheres,
                force_cpu=force_cpu,
                tol=tol, clamp_value=clamp_value,
                parallel=parallel, verbose=verbose)
        else:
            P_neg,N_neg,f_neg = None,None,None

        if P_pos is None or P_pos.size==0:
            P,N,f = P_neg,N_neg,neg[f_neg]
        elif P_neg is None or P_neg.size==0:
            P,N,f = P_pos,N_pos,pos[f_pos]
        else:
            P = np.concatenate((P_pos, P_neg), axis=0)
            N = np.concatenate((N_pos, N_neg), axis=0)
            f = np.concatenate((pos[f_pos], neg[f_neg]), axis=0)
    else:
        P,N,f = _sdf_to_point_cloud(U, S,
            rng_seed=seed(),
            rasterization_resolution=rasterization_resolution,
            n_local_searches=n_local_searches,
            local_search_iters=local_search_iters,
            batch_size=batch_size,
            force_cpu=force_cpu,
            tol=tol, clamp_value=clamp_value,
            parallel=parallel, verbose=verbose)

    if P is None or P.size==0:
        if verbose:
            print(f"Unable to find any point cloud point.")
        if return_point_cloud:
            return V, F, P, N
        else:
            return V, F

    if verbose:
        print(f"SDF to point cloud took {time.time()-t0_sdf_to_point_cloud}s")
        print("")

    if verbose:
        print(f"Fine tuning point cloud...")
        t0_fine_tune = time.time()

    P,N,f = _fine_tune_point_cloud(U, S, P, N, f,
        rng_seed=seed(),
        fine_tune_iters=fine_tune_iters,
        batch_size=batch_size,
        screening_weight=screening_weight,
        max_points_per_sphere=max_points_per_sphere,
        n_local_searches=n_local_searches,
        local_search_iters=local_search_iters,
        local_search_t=local_search_t,
        tol=tol,clamp_value=clamp_value,
        parallel=parallel, verbose=verbose)

    if verbose:
        print(f"Fine tuning point cloud took {time.time()-t0_fine_tune}s")
        print("")

    if verbose:
        print(f"Converting point cloud to mesh...")
        t0_point_cloud_to_mesh = time.time()

    V,F = point_cloud_to_mesh(P, N,
        method='PSR',
        psr_screening_weight=screening_weight,
        psr_outer_boundary_type="Neumann",
        verbose=False)

    if V is None or V.size==0:
        if verbose:
            print(f"Reconstructing point cloud failed.")
        if return_point_cloud: 
            if P is None or P.size==0:
                return V, F, P, N
            else:
                return V, F, scale*P+trans[None,:], N
        else:
            return V, F

    if verbose:
        print(f"Converting point cloud to mesh took {time.time()-t0_point_cloud_to_mesh}s")
        print("")

    if verbose:
        print(f" --- Finished Reach for the Arcs --- ")
        print(f"Total elapsed time: {time.time()-t0_total}s")

    # Undo recentering
    V = scale*V + trans[None,:]
    P = scale*P + trans[None,:]

    if return_point_cloud:
        return V, F, P, N
    else:
        return V, F


def _sdf_to_point_cloud(U, S,
    rng_seed=3452,
    rasterization_resolution=None,
    n_local_searches=None,
    local_search_iters=20,
    batch_size=10000,
    num_rasterization_spheres=0,
    tol=1e-4,
    clamp_value=np.inf,
    force_cpu=False,
    parallel=False,
    verbose=False):
    """Converts an SDF to a point cloud where all points are valid with respect
    to the spheres, and most spheres should have a tangent point.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    rasterization_resolution : int, optional (default 8*n_sdf**(1/d))
        the resolution of the rasterization grid
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere in
        the locally make feasible step
    batch_size : int, optional (default 10000)
        how many points in one batch. Set to 0 to disable batching.
    num_rasterization_spheres : int, optional (default 0)
        how many spheres to use at most in the rasterization step.
        Set to zero to use all spheres.
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    clamp_value : float, optional (default np.inf)
        value to which the SDF is clamped for clamped SDF reconstruction
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    f : (n_p,) numpy array
        feasible indices that map P back to U
    """

    assert np.min(U)>=0. and np.max(U)<=1.

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if rasterization_resolution is None:
        rasterization_resolution = 64 * math.ceil(n_sdf**(1./d)/16.)
        assert rasterization_resolution>0
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    if verbose:
        print(f"  Rasterization...")
        t0_rasterization = time.time()

    P = _outside_points_from_rasterization(U, S,
        rng_seed=seed(),
        res=rasterization_resolution, num_spheres=num_rasterization_spheres,
        tol=tol, force_cpu=force_cpu, parallel=parallel, verbose=verbose)

    if verbose:
        print(f"  Rasterization took {time.time()-t0_rasterization}s")
        print("")

    # If we found no points at all, return empty arrays here.
    if P.size == 0:
        return np.array([], dtype=np.float64), \
            np.array([], dtype=np.float64), \
            np.array([], dtype=np.int32)

    if verbose:
        print(f"  Locally make feasible...")
        t0_locally_make_feasible = time.time()

    P, N, f = _locally_make_feasible(U, S, P,
        rng_seed=seed(),
        n_local_searches=n_local_searches,
        local_search_iters=local_search_iters,
        batch_size=batch_size,
        tol=tol, clamp_value=clamp_value,
        parallel=parallel, verbose=verbose)

    if verbose:
        print(f"  Locally make feasible took {time.time()-t0_locally_make_feasible}s")
        print("")

    return P, N, f


def _outside_points_from_rasterization(U, S,
    rng_seed=3452,
    res=None,
    num_spheres=0,
    tol=1e-4,
    narrow_band=True,
    parallel=False,
    force_cpu=False,
    verbose=False):
    """Converts an SDF to a point cloud where all points should be outside our
    spheres, using rasterization.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    res : int, optional (default 8*n_sdf**(1/d))
        the resolution of the rasterization grid
    num_spheres : int, optional (default 0)
        how many spheres to use at most in the rasterization step.
        Set to zero to use all spheres.
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    narrow_band : bool, optional (default True)
        return points only in a narrow band around the spheres
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    force_cpu : bool, optional (default False)
        By default, the code chooses whether it wants to use the GPU or CPU
        based on heuristics. Set this to true to force CPU.
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        outside points
    """

    try:
        from gpytoolbox_bindings import _outside_points_from_rasterization_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    assert np.min(U)>=0. and np.max(U)<=1.

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if res is None:
        res = 64 * math.ceil(n_sdf**(1./d)/16.)
    # Maximum resolution so your GPU does not run out of memory.
    res = min(1024, res)

    assert res>=2, "Grid must have at least resolution 2."

    # Make sure res is divisible by 2.
    if res%2 != 0:
        res += 1

    if num_spheres > 0 and num_spheres < n_sdf:
        # Sample random indices to U and S
        rng = np.random.default_rng(rng_seed)
        # print(f"Sampling {num_spheres} random spheres from {n_sdf} sdf points.")
        indices = rng.choice(n_sdf, num_spheres, replace=False)
        U = U[indices,:]
        S = S[indices]

    P = _outside_points_from_rasterization_cpp_impl(U.astype(np.float64),
        np.abs(S).astype(np.float64),
        rng_seed, res, tol,
        narrow_band,
        parallel,
        force_cpu,
        verbose)

    return P



def _locally_make_feasible(U, S, P,
    rng_seed=3452,
    n_local_searches=None,
    local_search_iters=20,
    batch_size=10000,
    tol=1e-4,
    clamp_value=np.inf,
    parallel=False,
    verbose=False):
    """Given a number of SDF samples and points, tries to make each point
    feasible, and returns a list of feasible points at the end.

    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    P : (n_p,d) numpy array
        sampled points outside the spheres
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere
    batch_size : int, optional (default 1000)
        how many points in one batch. Set to 0 to disable batching.
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    clamp_value : float, optional (default np.inf)
        value to which the SDF is clamped for clamped SDF reconstruction
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    f : (n_p,) numpy array
        feasible indices that map P back to U
    """

    try:
        from gpytoolbox_bindings import _locally_make_feasible_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    assert np.min(U)>=0. and np.max(U)<=1.
    assert P.size > 0, "There needs to be at least one point outside the spheres."

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    # Batching
    if batch_size > 0 and batch_size < n_sdf:
        batch = rng.choice(n_sdf, batch_size)
    else:
        batch = np.arange(n_sdf)
        rng.shuffle(batch)

    P, N, f = _locally_make_feasible_cpp_impl(U.astype(np.float64),
        S.astype(np.float64), P.astype(np.float64),
        batch.astype(np.int32),
        seed(), 
        n_local_searches, local_search_iters,
        tol, clamp_value, parallel, verbose)

    if verbose:
        print(f"    {f.size} / {U.shape[0]} points are feasible.")

    return P, N, f


def _fine_tune_point_cloud(U, S, P, N, f,
    rng_seed=3452,
    fine_tune_iters=10,
    batch_size=10000,
    screening_weight=10.,
    max_points_per_sphere=3,
    n_local_searches=None,
    local_search_iters=20,
    local_search_t=0.01,
    tol=1e-4,
    clamp_value=np.inf,
    parallel=False,
    verbose=False):
    """Improve the point cloud with respect to the SDF such that the
    reconstructed surface will fulfill all sphere conditions
    
    IMPORTANT: IF YOU CALL THIS, YOUR SDF POINTS MUST BE IN [0,1]^d

    Parameters
    ----------
    U : (n_sdf,d) numpy array
        points where the SDF is evaluated
    S : (n_sdf,) numpy array
        sdf(U)
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    f : (n_p,) numpy array
        feasible indices that map P back to U
    rng_seed : int, optional (default 3452)
        rng seed where random data is needed
    fine_tune_iters : int, optional (default False)
        how may iterations to fine tune for
    batch_size : int, optional (default 1000)
        how many points in one batch. Set to 0 to disable batching.
    screening_weight : float, optional (default 0.1)
        PSR screening weight
    max_points_per_sphere : int, optional (default 3)
        How many points should there be at most per sphere.
        Set it to 1 to not add any additional points to any sphere.
        Has to be at least 1 (this is a theoretical minimum).
        If set to larger than 1, spheres that the reconstructed surface
        intersects will have at most max_points_per_sphere added.
    n_local_searches : int, optional (default 2*n_sdf**(1/d))
        how many local searches to perform for each sphere in the locally make
        feasible step
    local_search_iters : int, optional (default 20)
        how many iterations to try for in the local search for each sphere in
        the locally make feasible step
    local_search_t : double, optional (default 5e-3)
        how far to move each point in the local search for each point in the
        locally make feasible step
    tol : float, optional (default 1e-4)
        tolerance for determining whether a point is inside a sphere
    clamp_value : float, optional (default np.inf)
        value to which the SDF is clamped for clamped SDF reconstruction
    parallel : bool, optional (default False)
        whether to parallelize the algorithm or not
    verbose : bool, optional (default False)
        whether to output details on the algorithm when it's running

    
    Returns
    -------
    P : (n_p,d) numpy array
        reconstructed point cloud points
    N : (n_p,d) numpy array
        reconstructed point cloud normals
    f : (n_p,) numpy array
        feasible indices that map P back to U
    """

    d = U.shape[1]
    assert d==2 or d==3, "Only dimensions 2 and 3 supported."
    assert max_points_per_sphere>=1, "There has to be at least one point per sphere."

    n_sdf = U.shape[0]

    # Pick default values if not supplied.
    if n_local_searches is None:
        n_local_searches = math.ceil(2. * n_sdf**(1./d))

    # RNG used to compute random numbers and new seeds during the method.
    rng = np.random.default_rng(seed=rng_seed)
    seed = lambda : rng.integers(0,np.iinfo(np.int32).max)

    if verbose:
        print(f"  Fine tune called with {f.size} / {U.shape[0]} feasible points.")

    for it in range(fine_tune_iters):
        #Generate a random batch of size batch size.
        if batch_size > 0 and batch_size < n_sdf:
            batch = rng.choice(n_sdf, batch_size)
        else:
            batch = np.arange(n_sdf)
            rng.shuffle(batch)

        V,F = point_cloud_to_mesh(P, N,
            psr_screening_weight=screening_weight,
            psr_outer_boundary_type="Neumann",
            verbose=False)

        if(V.size == 0):
            if(verbose):
                print(f"    point_cloud_to_mesh did not produce a mesh.")
            return P, N, f

        P, N, f = _fine_tune_point_cloud_iteration(U,
            S,
            V,
            F,
            P,
            N,
            f,
            batch,
            max_points_per_sphere,
            seed(),
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, clamp_value, parallel, verbose)

        if verbose:
            print(f"  After fine tuning iter {it}, we have {f.size} points.")
    return P, N, f


def _fine_tune_point_cloud_iteration(U, S, 
    V, F,
    P, N, f,
    batch,
    max_points_per_sphere,
    rng_seed,
    n_local_searches,
    local_search_iters,
    local_search_t,
    tol, clamp_value,
    parallel,
    verbose):
    """Even if you batch, please pass the entirety of of U, S to this function.
    """

    try:
        from gpytoolbox_bindings import _fine_tune_point_cloud_iter_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    P, N, f = _fine_tune_point_cloud_iter_cpp_impl(U.astype(np.float64),
            S.astype(np.float64),
            V.astype(np.float64),
            F.astype(np.int32),
            P.astype(np.float64),
            N.astype(np.float64),
            f.astype(np.int32),
            batch.astype(np.int32),
            max_points_per_sphere,
            rng_seed,
            n_local_searches,
            local_search_iters,
            local_search_t,
            tol, clamp_value, parallel, verbose)

    return P, N, f


