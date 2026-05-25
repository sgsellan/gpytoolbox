import numpy as np
from .particle_swarm import particle_swarm
from .signed_distance import signed_distance
from .ray_mesh_intersect import ray_mesh_intersect
from .random_points_on_mesh import random_points_on_mesh


def _axis_angle_to_matrix(aa):
    """Rodrigues' formula: 3-vector → 3x3 rotation matrix."""
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        # Fall back to +z if degenerate.
        out = np.zeros_like(v)
        out[2] = 1.0
        return out
    return v / n


def _transform_points(B, s, R, c, B_center):
    """T(x) = c + s * R * (x - B_center)."""
    return (s * (B - B_center) @ R.T) + c


def _feasible(samples_TB, A_V, A_F,
              cut_point, cut_normal,
              a_plus, a_minus,
              sd_tol=1e-6, plane_tol=1e-6):
    """Test if a configuration of T(B) sample points nests inside A and the two
    halves of A can clear T(B) along a+/a-.

    samples_TB : (m,3) — points on the (transformed) surface of B.
    cut_normal : unit vector defining the cut plane.
    a_plus     : removal direction for A_top (above the plane). a+ · n > 0.
    a_minus    : removal direction for A_bot. a- · n < 0.
    """
    # 1) Containment: every surface sample of T(B) is strictly inside A
    #    (signed_distance < 0 inside, > 0 outside).
    sd, _, _ = signed_distance(samples_TB, A_V, A_F)
    if np.any(sd > -sd_tol):
        return False

    # 2) Removal test for A_top along a+ : cast rays from every T(B) sample
    #    in direction -a+. If any hits A's boundary strictly above the cut
    #    plane, A_top cannot clear T(B) as it slides along a+.
    m = samples_TB.shape[0]
    a_plus_n = _normalize(a_plus)
    a_minus_n = _normalize(a_minus)
    dirs_plus = np.tile(-a_plus_n[None, :], (m, 1))
    dirs_minus = np.tile(-a_minus_n[None, :], (m, 1))

    ts, ids, _ = ray_mesh_intersect(samples_TB, dirs_plus, A_V, A_F)
    if np.any(ids >= 0):
        hit_mask = ids >= 0
        hits = samples_TB[hit_mask] + ts[hit_mask, None] * dirs_plus[hit_mask]
        side = (hits - cut_point) @ cut_normal
        if np.any(side > plane_tol):
            return False

    ts, ids, _ = ray_mesh_intersect(samples_TB, dirs_minus, A_V, A_F)
    if np.any(ids >= 0):
        hit_mask = ids >= 0
        hits = samples_TB[hit_mask] + ts[hit_mask, None] * dirs_minus[hit_mask]
        side = (hits - cut_point) @ cut_normal
        if np.any(side < -plane_tol):
            return False

    return True


def _sample_B_surface(B_V, B_F, n_samples, rng):
    """Sample n_samples points uniformly on B's surface. Always include the
    actual mesh vertices too — they are extreme points and matter for containment."""
    if n_samples <= 0:
        return B_V.copy()
    pts = random_points_on_mesh(B_V, B_F, n_samples, rng=rng)
    return np.vstack([B_V, pts])


def _largest_feasible_scale(samples_B, B_center, A_V, A_F, R, c,
                            cut_point, cut_normal, a_plus, a_minus,
                            s_lo=0.0, s_hi=1.0, tol=1e-3, max_iter=15):
    """Binary-search the largest s in [s_lo, s_hi] for which the nesting is feasible.

    Assumes s = s_lo is feasible (a tiny shape inside A is always feasible if
    centered inside A). Returns 0.0 if even s_lo is infeasible.
    """
    def feas(s):
        samples_TB = _transform_points(samples_B, s, R, c, B_center)
        return _feasible(samples_TB, A_V, A_F, cut_point, cut_normal,
                         a_plus, a_minus)

    # Sanity check the lower bound; if even small scale fails, the centroid is
    # outside A (or some other invalid config) — give up.
    if not feas(s_lo if s_lo > 0 else 1e-3):
        return 0.0

    lo, hi = max(s_lo, 1e-3), s_hi
    # If the upper bound is already feasible, return it.
    if feas(hi):
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if feas(mid):
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return lo


def _matrix_to_axis_angle(R):
    """Inverse of Rodrigues' formula. Returns a 3-vector whose direction is
    the axis and whose magnitude is the angle in [0, pi]."""
    # Numerically robust extraction.
    cos_theta = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near-pi case: extract axis from the symmetric part.
        M = 0.5 * (R + np.eye(3))
        # Pick the column with the largest diagonal as the axis estimate.
        diag = np.diag(M)
        i = int(np.argmax(diag))
        axis = M[:, i] / np.sqrt(max(diag[i], 1e-12))
        return theta * axis / np.linalg.norm(axis)
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return theta * k


def _encode_all(R, c, cut_normal, cut_point, a_plus, a_minus, cut_anchor):
    """Pack (R, c, plane, removal dirs) into the 16-D 'all'-mode vector that
    `_decode` would map back to the same configuration (up to sign convention)."""
    aa = _matrix_to_axis_angle(R)
    cn = _normalize(cut_normal)
    offset = float(np.dot(cut_point - cut_anchor, cn))
    ap = _normalize(a_plus)
    if np.dot(ap, cn) < 0:
        ap = -ap
    am = _normalize(a_minus)
    if np.dot(am, cn) > 0:
        am = -am
    return np.concatenate([aa, c, cn, [offset], ap, am])


def _decode(x, mode, fixed):
    """Map a flat parameter vector x to (R, c, cut_point, cut_normal, a_plus, a_minus).

    mode: 'rigid' or 'all'.
    fixed: dict with any pre-set values (used for variables not in x).
    """
    i = 0
    aa = x[i:i + 3]; i += 3
    c = x[i:i + 3]; i += 3
    R = _axis_angle_to_matrix(aa)

    if mode == 'all':
        # 3 params for cut normal direction (then normalized), 1 for offset
        n_raw = x[i:i + 3]; i += 3
        cut_normal = _normalize(n_raw)
        offset = x[i]; i += 1
        cut_point = fixed['cut_anchor'] + offset * cut_normal

        # 3 params each for a+ and a- (then normalized, sign aligned with n)
        ap_raw = x[i:i + 3]; i += 3
        am_raw = x[i:i + 3]; i += 3
        a_plus = _normalize(ap_raw)
        if np.dot(a_plus, cut_normal) < 0:
            a_plus = -a_plus
        a_minus = _normalize(am_raw)
        if np.dot(a_minus, cut_normal) > 0:
            a_minus = -a_minus
    else:
        cut_point = fixed['cut_point']
        cut_normal = fixed['cut_normal']
        a_plus = fixed['a_plus']
        a_minus = fixed['a_minus']

    return R, c, cut_point, cut_normal, a_plus, a_minus


def matryoshka(V, F,
               VB=None, FB=None,
               optimize='all',
               R=None, c=None,
               cut_point=None, cut_normal=None,
               a_plus=None, a_minus=None,
               n_samples=200,
               n_particles=30, max_iter=40,
               scale_tol=1e-3,
               warm_start=False,
               verbose=False,
               seed=None):
    """Generalized Matryoshka: find a similarity transform of `B` that nests
    inside `A` such that `A` can be cut by a plane and pulled apart along
    `a+`/`a-` without colliding with the inner copy. Implements the algorithm
    of Jacobson (SGP 2017) on the CPU using particle swarm optimization.

    Parameters
    ----------
    V : (n,3) numpy double array
        Vertex positions of the outer mesh A.
    F : (m,3) numpy int array
        Triangle indices of A.
    VB, FB : optional inner mesh B. If None, self-nesting is performed (B = A).
    optimize : str, optional (default 'all')
        Which variables to optimize:
        * 'all'        — scale, rotation, centroid, cut plane, removal directions
        * 'rigid'      — scale, rotation, centroid (cut plane and removal directions fixed)
        * 'scale_only' — only the scale (everything else must be provided)
    R, c, cut_point, cut_normal, a_plus, a_minus : optional fixed values used
        either as defaults for non-optimized variables or, in 'scale_only' mode,
        as the entire configuration. If `cut_point` / `cut_normal` are not
        provided, a horizontal cut through the centroid of A is used.
    n_samples : int, optional (default 200)
        Number of random surface samples drawn from B (in addition to B's
        vertices) for the feasibility test.
    n_particles, max_iter : int, optional
        Particle swarm hyperparameters (kept small by default since each
        feasibility evaluation is expensive on the CPU).
    scale_tol : float, optional (default 1e-3)
        Binary-search tolerance for the inner scale search.
    warm_start : bool or dict, optional (default False)
        Only relevant for `optimize='all'`. If True, run a `'rigid'`
        optimization first (with the same budget) and use its solution to
        seed the global-best tracker of the full optimization. Alternatively,
        pass a result dict from a previous `matryoshka(...)` call to use
        that as the seed (useful when you want to compare against a
        specific baseline). Either form guarantees that the returned scale
        is at least as large as the seed scale (modulo `scale_tol`).
    verbose : bool, optional (default False)
        Print particle-swarm progress.
    seed : int or None, optional
        Seed for the surface-sampling RNG and the particle-swarm RNG.

    Returns
    -------
    result : dict with keys
        s          : float, the optimal scale.
        R          : (3,3) rotation matrix applied to B.
        c          : (3,) translation, the new centroid of T(B).
        B_center   : (3,) original centroid of B used as the rotation pivot.
        cut_point  : (3,) point on the cut plane.
        cut_normal : (3,) unit normal of the cut plane.
        a_plus     : (3,) removal direction for A above the plane.
        a_minus    : (3,) removal direction for A below the plane.

    Notes
    -----
    This is a CPU implementation of a method originally formulated with GPU
    depth peeling. It is therefore much slower than the original. Use modest
    `n_samples`, `n_particles` and `max_iter` for interactive experimentation.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    if VB is None:
        VB = V
        FB = F
    VB = np.asarray(VB, dtype=np.float64)
    FB = np.asarray(FB, dtype=np.int32)

    rng = np.random.default_rng(seed)

    B_center = VB.mean(axis=0)
    A_center = V.mean(axis=0)
    A_min = V.min(axis=0)
    A_max = V.max(axis=0)
    A_extent = A_max - A_min

    # Defaults for fixed values (used when optimize != 'all' or values not
    # provided).
    if cut_normal is None:
        cut_normal = np.array([0.0, 0.0, 1.0])
    else:
        cut_normal = _normalize(np.asarray(cut_normal, dtype=np.float64))
    if cut_point is None:
        cut_point = A_center.copy()
    else:
        cut_point = np.asarray(cut_point, dtype=np.float64)
    if a_plus is None:
        a_plus = cut_normal.copy()
    else:
        a_plus = _normalize(np.asarray(a_plus, dtype=np.float64))
    if a_minus is None:
        a_minus = -cut_normal.copy()
    else:
        a_minus = _normalize(np.asarray(a_minus, dtype=np.float64))

    # Surface samples of B (canonical, before transformation).
    samples_B = _sample_B_surface(VB, FB, n_samples, rng)

    if optimize == 'scale_only':
        if R is None:
            R = np.eye(3)
        if c is None:
            c = A_center.copy()
        s = _largest_feasible_scale(samples_B, B_center, V, F, R, c,
                                    cut_point, cut_normal, a_plus, a_minus,
                                    tol=scale_tol)
        return dict(s=s, R=R, c=c, B_center=B_center,
                    cut_point=cut_point, cut_normal=cut_normal,
                    a_plus=a_plus, a_minus=a_minus)

    # Build (lb, ub) for the free parameters.
    # Always-free: 3 axis-angle + 3 centroid
    lb = [-np.pi, -np.pi, -np.pi,
          A_min[0], A_min[1], A_min[2]]
    ub = [np.pi, np.pi, np.pi,
          A_max[0], A_max[1], A_max[2]]
    fixed = dict(cut_point=cut_point, cut_normal=cut_normal,
                 a_plus=a_plus, a_minus=a_minus,
                 cut_anchor=A_center.copy())

    if optimize == 'all':
        # cut normal direction (3 free), offset along normal (1), a+ (3), a- (3)
        max_offset = 0.5 * np.linalg.norm(A_extent)
        lb += [-1.0, -1.0, -1.0, -max_offset, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        ub += [1.0, 1.0, 1.0, max_offset, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    lb = np.array(lb, dtype=np.float64)
    ub = np.array(ub, dtype=np.float64)

    # Use a deterministic rng feeding into particle_swarm via numpy global seed
    # (particle_swarm uses np.random internally for the Python path; the C++
    # path uses a non-seeded RNG, which is fine — the objective itself does
    # not depend on the swarm's RNG for reproducibility of the *result*).
    if seed is not None:
        np.random.seed(seed)

    # Track the best configuration encountered (the swarm only tracks the
    # best objective; we need the full configuration to return).
    best = {'s': 0.0, 'x': None}

    # Optional warm-start: seed the global-best tracker with a previously
    # computed configuration. The swarm itself starts from random particles,
    # but our external `best` tracker retains the seed whenever the swarm
    # doesn't beat it. The seed can be either a result dict from a prior
    # `matryoshka(...)` call, or True (run 'rigid' internally first).
    if warm_start and optimize == 'all':
        if isinstance(warm_start, dict):
            seed_res = warm_start
        else:
            seed_res = matryoshka(
                V, F, VB=VB, FB=FB, optimize='rigid',
                R=R, c=c,
                cut_point=cut_point, cut_normal=cut_normal,
                a_plus=a_plus, a_minus=a_minus,
                n_samples=n_samples,
                n_particles=n_particles, max_iter=max_iter,
                scale_tol=scale_tol, verbose=verbose, seed=seed)
        if seed_res['s'] > 0:
            seed_x = _encode_all(
                seed_res['R'], seed_res['c'],
                seed_res['cut_normal'], seed_res['cut_point'],
                seed_res['a_plus'], seed_res['a_minus'],
                fixed['cut_anchor'])
            # Re-evaluate the seed's scale with the outer call's samples_B,
            # which may differ from whatever sampling produced seed_res. This
            # keeps best['s'] self-consistent: it is always the maximum
            # feasible scale of best['x'] *as measured here*.
            R_s, c_s, cp_s, cn_s, ap_s, am_s = _decode(seed_x, 'all', fixed)
            best['s'] = _largest_feasible_scale(
                samples_B, B_center, V, F, R_s, c_s,
                cp_s, cn_s, ap_s, am_s, tol=scale_tol)
            best['x'] = seed_x

    def objective(x):
        R_x, c_x, cp, cn, ap, am = _decode(x, optimize, fixed)
        s = _largest_feasible_scale(samples_B, B_center, V, F, R_x, c_x,
                                    cp, cn, ap, am, tol=scale_tol)
        if s > best['s']:
            best['s'] = s
            best['x'] = x.copy()
        # We minimize: negative scale.
        return -s

    _x_best, _f_best = particle_swarm(
        objective, lb, ub,
        n_particles=n_particles, max_iter=max_iter,
        verbose=verbose, topology='full')

    if best['x'] is None:
        # Swarm never found a feasible scale; return zero-scale result.
        return dict(s=0.0,
                    R=np.eye(3), c=A_center.copy(), B_center=B_center,
                    cut_point=cut_point, cut_normal=cut_normal,
                    a_plus=a_plus, a_minus=a_minus)

    R_out, c_out, cp_out, cn_out, ap_out, am_out = _decode(
        best['x'], optimize, fixed)
    return dict(s=best['s'], R=R_out, c=c_out, B_center=B_center,
                cut_point=cp_out, cut_normal=cn_out,
                a_plus=ap_out, a_minus=am_out)
