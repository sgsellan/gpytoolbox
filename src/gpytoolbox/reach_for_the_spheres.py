# Here I import only the functions I need for these functions
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, block_diag, diags
from scipy.spatial import cKDTree
from .squared_distance import squared_distance
from .fixed_dof_solve   import fixed_dof_solve, fixed_dof_solve_precompute
from .boundary_vertices import boundary_vertices
from .boundary_faces    import boundary_faces
from .remove_unreferenced import remove_unreferenced
from .remove_duplicate_vertices import remove_duplicate_vertices
from .triangle_triangle_adjacency import triangle_triangle_adjacency
from .grad import grad
from .massmatrix_intrinsic import massmatrix_intrinsic
from .cotangent_laplacian import cotangent_laplacian
from .doublearea import doublearea
from .per_face_normals import per_face_normals
from .tip_angles import tip_angles
from .halfedge_lengths_squared import halfedge_lengths_squared
from .remesh_botsch import remesh_botsch
from .random_points_on_mesh import random_points_on_mesh
from .non_manifold_edges import non_manifold_edges

def reach_for_the_spheres(U, sdf, V, F, S=None,
    return_U=False,
    verbose=False,
    max_iter=None, tol=None, h=None, min_h=None,
    linesearch=None, min_t=None, max_t=None,
    dt=None,
    inside_outside_test=None,
    resample=None,
    resample_samples=None,
    feature_detection=None,
    output_sensitive=None,
    remesh_iterations=None,
    batch_size=None,
    fix_boundary=None,
    clamp=None, pseudosdf_interior=None):
    """Creates a mesh from a signed distance function (SDF) using the
    "Reach for the Spheres" method of S. Sellán,  C. Batty, and O. Stein [2023].

    This method takes in an sdf, sample points (do not need to be on a grid),
    and an initial mesh.
    It then flows this initial mesh into a reconstructed mesh that fulfills
    the signed distance function.

    This method works in dimension 2 (dim==2), where it reconstructs a polyline,
    and in dimension 3 (dim==3), where it reconstructs a triangle mesh.
    
    Parameters
    ----------
    U : (k,dim) numpy double array
        Matrix of SDF sample positions.
        The sdf will be sampled at these points
    sdf: lambda function that takes in a (ks,dim) numpy double array
        The signed distance function to be reconstructed.
        This function must take in a (ks,dim) matrix of sample points and
        return a (ks,) array of SDF values at these points.
    V : (n,dim) numpy double array
        Matrix of mesh vertices of the initial mesh
    F : (m,dim) numpy int array
        Matrix of triangle indices into V of the initlal mesh
    S : (k,dim) nummpy double array, optional (default: None)
        Matrix of SDF samples at the sample positions in U.
        If this is not provided, it will be computed using sdf.
    return_U : bool, optional (default: False)
        Whether to return the matrix of SDF sample positions along with the
        reconstructed mesh.
    verbose : bool (default false)
        Whether to print method statistics during operation.
    max_iter : int (default None)
        The maximum number of iterations to perform for the method.
        If not supplied, a sensible default is used.
    tol : float (default None)
        The method's tolerance for the sphere tangency test.
        If not supplied, a sensible default is used.
    h : float (default None)
        The method's initial target mesh length for the reconstructed mesh.
        This will change during iteration, set min_h if you want to control the
        minimum edge length overall.
        If not supplied, a sensible default is used.
    min_h : float (default None)
        The method's minimal target edge length for the reconstructed mesh.
        If not supplied, a sensible default is used.
    linesearch : bool (default None)
        Whether to use a linesearch-like heuristic for the method's timestep.
        If not supplied, linesearch is used.
    min_t : float (default None)
        The method's minimum timestep.
        If not supplied, a sensible default is used.
    max_t : float (default None)
        The method's minimum timestep.
        If not supplied, a sensible default is used.
    dt : float (default None)
        The method's default timestep.
        If not supplied, a sensible default is used.
    inside_outside_test : bool (default None)
        Whether to use inside-outside testing when projecting points to be
        tangent to the sphere.
        Turn this off if your distance function is *unsigned*.
        If not supplied, inside-outside test is used
    resample : int (default None)
        How often to resample the SDF after convergence to extract more
        information.
        If not supplied, resampling is not performed.
    resample_samples : int (default None)
        How many samples to use when resampling.
        If not supplied, a sensible default is used.
    feature_detection : string (default None)
        Which feature detection mode to use.
        If not supplied, aggressive feature detection is used.
    output_sensitive : bool (default None)
        Whether to use output-sensitive remeshing.
        If not supplied, remeshing is output-sensitive.
    remesh_iterations : int (default None)
        How many iterations of the remesher to run each step.
        If not supplied, a sensible default is used.
    batch_size : int (default None)
        For large amounts of sample points, the method is sped up using sample
        point batching.
        This parameter specifies how many samples to take for each batch.
        Set it to 0 to disable batching.
        If not supplied, a sensible default is used.
    fix_boundary : bool (default None)
        Whether to fix the boundary of the mesh during iteration.
        If not supplied, the boundary is not fixed.
    clamp : float (default None)
        If sdf is a clamped SDF, the clamp value to use.
        np.inf for no clamping.
        If not supplied, there is no clamping.
    pseudosdf_interior : bool (default None)
        If enabled, treats every negative SDF value as a bound on the signed
        distance, as opposed to an exact signed distance, for use in SDFs
        resulting from CSG unions as described by Marschner et al.
        "Constructive Solid Geometry on Neural Signed Distance Fields" [2023].
        If not supplied, this feature is disabled.
        
    Returns
    -------
    Vr : (nr,dim) numpy double array
        Matrix of mesh vertices of the reconstructed triangle mesh
    Fr : (mr,dim) numpy int array
        Matrix of triangle indices into Vr of the reconstructed mesh
    Ur : (kr,dim) numpy double array, if requested
        Matrix of SDF sample positions.
        This can be different from the supplied Ur if the method is set to
        resample.
    
    See Also
    --------
    marching_squares, marching_cubes

    Notes
    --------
    This method has a number of limitations that are described in the paper.
    E.g., the method will only work for SDFs that describe surfaces with the
    same topology as the initial surface.

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    import numpy as npy

    # Get a signed distance function
    V,F = gpy.read_mesh("my_mesh.obj")

    # Create an SDF for the mesh
    j = 20
    sdf = lambda x: gpy.signed_distance(x, V, F)[0]
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T

    # Choose an initial surface for reach_for_the_spheres
    V0, F0 = gpy.icosphere(2)

    # Reconstruct triangle mesh
    Vr,Fr = gpy.reach_for_the_spheres(U, sdf, V0, F0)

    #The reconstructed triangle mesh is now Vr,Fr.
    ```
    """


    state = ReachForTheSpheresState(V=V, F=F, sdf=sdf, U=U, S=S,
        h=h, min_h=min_h)
    converged = False

    while not converged:
        converged = reach_for_the_spheres_iteration(state,
            max_iter=max_iter, tol=tol,
            linesearch=linesearch, min_t=min_t, max_t=max_t,
            dt=dt,
            inside_outside_test=inside_outside_test,
            resample=resample,
            resample_samples=resample_samples,
            feature_detection=feature_detection,
            output_sensitive=output_sensitive,
            remesh_iterations=remesh_iterations,
            batch_size=batch_size,
            verbose=verbose,
            fix_boundary=fix_boundary,
            clamp=clamp, pseudosdf_interior=pseudosdf_interior)

    if return_U:
        return state.V, state.F, state.U
    else:
        return state.V, state.F


class ReachForTheSpheresState:
    """An object to keep state during the iterations of the Reach for the
    Spheres method.
    Meant to be used in conjunction with `reach_for_the_spheres_iteration`.
    
    Parameters
    ----------
    V : (n,dim) numpy double array
        Matrix of mesh vertices of the initial mesh
    F : (m,dim) numpy int array
        Matrix of triangle indices into V of the initlal mesh
    U : (k,dim) numpy double array, optional (default: None)
        Matrix of SDF sample positions.
        The sdf will be sampled at these points
    S : (k,dim) nummpy double array, optional (default: None)
        Matrix of SDF samples at the sample positions in U.
        If this is not provided, it will be computed using sdf.
    sdf: lambda function that takes in a (ks,dim) numpy double array, optional (default: None)
        The signed distance function to be reconstructed.
        This function must take in a (ks,dim) matrix of sample points and
        return a (ks,) array of SDF values at these points.
    V_active : (na,dim) numpy double array, optional (default: None)
        Matrix of mesh vertices active during iteration
    F_active : (ma,dim) numpy int array, optional (default: None)
        Matrix of triangle indices into V_active of the mesh active during iteration
    V_inactive : (na,dim) numpy double array, optional (default: None)
        Matrix of mesh vertices inactive during iteration
    F_inactive : (ma,dim) numpy int array, optional (default: None)
        Matrix of triangle indices into V_inactive of the mesh inactive during iteration
    rng : numpy random generator, optional (default: None)
        numpy rng object used to create randomness during iterations
    h : float (default None)
        The method's initial target mesh length for the reconstructed mesh.
        This will change during iteration, set min_h if you want to control the
        minimum edge length overall.
    min_h : float (default None)
        The method's minimal target edge length for the reconstructed mesh.
    best_performance : float (default None)
        Remembers the method's best performance
    convergence_counter : int (default None)
        The method's convergence counter, used to track for how long progress
        has not been made.
    best_avg_error : float (default None)
        Remembers the method's average error.
    resample_counter : int (default None)
        Tracks how often the SDF has been resampled.
    V_last_converged : (nl,dim) numpy double array, optional (default: None)
        Matrix of mesh vertices for the last known converged mesh.
    F_last_converged : (ml,dim) numpy int array, optional (default: None)
        Matrix of triangle indices into V_last_converged for the last known
        converged mesh.
    U_batch : (kb,dim) numpy double array, optional (default: None)
        Matrix of SDF sample positions as used during the last batching
        operation.
    S_batch : (kb,dim) nummpy double array, optional (default: None)
        Matrix of SDF samples at the sample positions in U as used during the
        last batching operation.

    Notes
    --------
    If you create this object yourself, you should supply V, F, U, S, sdf.
    Supply all other parameters only as explicitly needed.

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    import numpy as npy

    # Get a signed distance function
    V,F = gpy.read_mesh("my_mesh.obj")

    # Create an SDF for the mesh
    j = 20
    sdf = lambda x: gpy.signed_distance(x, V, F)[0]
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T

    # Create ReachForTheSpheresState to use in iterative method later
    V0, F0 = gpy.icosphere(2)
    state = ReachForTheSpheresState(V=V0, F=F0, sdf=sdf, U=U)

    # Run one iteration of Reach for the Spheres
    converged = gpy.reach_for_the_spheres_iteration(state)

    # Reconstruct triangle mesh
    Vr,Fr = state.V,state.F

    #The reconstructed mesh after one iteration is now Vr,Fr
    ```
    """

    def __init__(self,
        V, F,
        U=None,
        S=None,
        sdf=None,
        V_active=None, F_active=None,
        V_inactive=None, F_inactive=None,
        rng=None,
        h=None, min_h=None, its=None,
        best_performance=None,
        convergence_counter=None,
        best_avg_error=None,
        resample_counter=None,
        V_last_converged=None, F_last_converged=None,
        U_batch=None, S_batch=None
        ):

        self.V=V
        self.F=F
        self.U=U
        self.S=S
        self.sdf=sdf
        self.V_active = V_active
        self.F_active = F_active
        self.V_inactive = V_inactive
        self.F_inactive = F_inactive
        self.rng = rng
        self.h = h
        self.min_h = min_h
        self.its = its
        self.best_performance = best_performance
        self.convergence_counter = convergence_counter
        self.best_avg_error = best_avg_error
        self.resample_counter = resample_counter
        self.V_last_converged = V_last_converged
        self.F_last_converged = F_last_converged
        self.U_batch = U_batch
        self.S_batch = S_batch


#Returns true if algorithm has converged.
def reach_for_the_spheres_iteration(state,
    max_iter=None, tol=None,
    linesearch=None, min_t=None, max_t=None,
    dt=None,
    inside_outside_test=None,
    resample=None,
    resample_samples=None,
    feature_detection=None,
    output_sensitive=None,
    remesh_iterations=None,
    batch_size=None,
    fix_boundary=None,
    clamp=None, pseudosdf_interior=None,
    verbose=False):
    """Performs one iteration of the "Reach for the Spheres" method of
    S. Sellán,  C. Batty, and O. Stein [2023].
    This method is used to create a mesh from a signed distance function (SDF).

    This method takes in the current state of the method in the form of a 
    ReachForTheSpheresState object, and stores its results as well as any
    temporary information needed in that state object.

    This method works in dimension 2 (dim==2), where it reconstructs a polyline,
    and in dimension 3 (dim==3), where it reconstructs a triangle mesh.
    
    Parameters
    ----------
    state: ReachForTheSpheresState object
        Stores all needed information about the current state of the method.
    max_iter : int (default None)
        The maximum number of iterations to perform for the method.
        If not supplied, a sensible default is used.
    tol : float (default None)
        The method's tolerance for the sphere tangency test.
        If not supplied, a sensible default is used.
    linesearch : bool (default None)
        Whether to use a linesearch-like heuristic for the method's timestep.
        If not supplied, linesearch is used.
    min_t : float (default None)
        The method's minimum timestep.
        If not supplied, a sensible default is used.
    max_t : float (default None)
        The method's minimum timestep.
        If not supplied, a sensible default is used.
    dt : float (default None)
        The method's default timestep.
        If not supplied, a sensible default is used.
    inside_outside_test : bool (default None)
        Whether to use inside-outside testing when projecting points to be
        tangent to the sphere.
        Turn this off if your distance function is *unsigned*.
        If not supplied, inside-outside test is used
    resample : int (default None)
        How often to resample the SDF after convergence to extract more
        information.
        If not supplied, resampling is not performed.
    resample_samples : int (default None)
        How many samples to use when resampling.
        If not supplied, a sensible default is used.
    feature_detection : string (default None)
        Which feature detection mode to use.
        If not supplied, aggressive feature detection is used.
    output_sensitive : bool (default None)
        Whether to use output-sensitive remeshing.
        If not supplied, remeshing is output-sensitive.
    remesh_iterations : int (default None)
        How many iterations of the remesher to run each step.
        If not supplied, a sensible default is used.
    batch_size : int (default None)
        For large amounts of sample points, the method is sped up using sample
        point batching.
        This parameter specifies how many samples to take for each batch.
        Set it to 0 to disable batching.
        If not supplied, a sensible default is used.
    fix_boundary : bool (default None)
        Whether to fix the boundary of the mesh during iteration.
        If not supplied, the boundary is not fixed.
    clamp : float (default None)
        If sdf is a clamped SDF, the clamp value to use.
        np.inf for no clamping.
        If not supplied, there is no clamping.
    pseudosdf_interior : bool (default None)
        If enabled, treats every negative SDF value as a bound on the signed
        distance, as opposed to an exact signed distance, for use in SDFs
        resulting from CSG unions as described by Marschner et al.
        "Constructive Solid Geometry on Neural Signed Distance Fields" [2023].
        If not supplied, this feature is disabled.
    verbose : bool (default false)
        Whether to print method statistics during operation.
        
    Returns
    -------
    converged : bool
        Whether the method has converged after this iteration or not.
    
    See Also
    --------
    reach_for_the_spheres, marching_squares, marching_cubes

    Notes
    --------
    This method has a number of limitations that are described in the paper.
    E.g., the method will only work for SDFs that describe surfaces with the
    same topology as the initial surface.

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    import numpy as npy

    # Get a signed distance function
    V,F = gpy.read_mesh("my_mesh.obj")

    # Create an SDF for the mesh
    j = 20
    sdf = lambda x: gpy.signed_distance(x, V, F)[0]
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1), np.linspace(-1.0, 1.0, j+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T

    # Create ReachForTheSpheresState to use in iterative method later
    V0, F0 = gpy.icosphere(2)
    state = ReachForTheSpheresState(V=V0, F=F0, sdf=sdf, U=U)

    # Run one iteration of Reach for the Spheres
    converged = gpy.reach_for_the_spheres_iteration(state)

    # Reconstruct triangle mesh
    Vr,Fr = state.V,state.F

    #The reconstructed mesh after one iteration is now Vr,Fr
    ```
    """


    assert isinstance(state, ReachForTheSpheresState), "State must be a ReachForTheSpheresState"
    dim = state.V.shape[1]
    assert dim==state.F.shape[1]
    if state.U is not None:
        assert dim==state.U.shape[1]

    #Set default parameters.
    #Assign different 2D and 3D default parameters
    default_params = {
        #For each dimension
        2: {'max_iter':10000, 'tol':5e-3, 'h':0.1,
        'linesearch':True, 'min_t':1e-6, 'max_t':50.,
        'dt':10.,
        'inside_outside_test':True,
        'resample':0, 'resample_samples':2*int(np.ceil(np.sqrt(state.V.shape[0] if state.U is None else state.U.shape[0]))),
        'return_U':False,
        'feature_detection':'aggressive', 'output_sensitive':True,
        'remesh_iterations':1,
        'batch_size':20000,
        'fix_boundary':False,
        'clamp':np.inf, 'pseudosdf_interior':False},
        3: {'max_iter':20000, 'tol':1e-2, 'h':0.2,
        'linesearch':True, 'min_t':1e-6, 'max_t':50.,
        'dt':10.,
        'inside_outside_test':True,
        'resample':0, 'resample_samples':2*int(np.ceil(np.cbrt(state.V.shape[0] if state.U is None else state.U.shape[0]))),
        'feature_detection':'aggressive', 'output_sensitive':True,
        'remesh_iterations':1,
        'visualize':False,
        'batch_size':20000,
        'fix_boundary':False,
        'clamp':np.inf, 'pseudosdf_interior':False}
    }
    if max_iter is None:
        max_iter = default_params[dim]['max_iter']
    if tol is None:
        tol = default_params[dim]['tol']
    if linesearch is None:
        linesearch = default_params[dim]['linesearch']
    if min_t is None:
        min_t = default_params[dim]['min_t']
    if max_t is None:
        max_t = default_params[dim]['max_t']
    if dt is None:
        dt = default_params[dim]['dt']
    if inside_outside_test is None:
        inside_outside_test = default_params[dim]['inside_outside_test']
    if resample is None:
        resample = default_params[dim]['resample']
    if resample_samples is None:
        resample_samples = default_params[dim]['resample_samples']
    if feature_detection is None:
        feature_detection = default_params[dim]['feature_detection']
    if output_sensitive is None:
        output_sensitive = default_params[dim]['output_sensitive']
    if remesh_iterations is None:
        remesh_iterations = default_params[dim]['remesh_iterations']
    if batch_size is None:
        batch_size = default_params[dim]['batch_size']
    if fix_boundary is None:
        fix_boundary = default_params[dim]['fix_boundary']
    if clamp is None:
        clamp = default_params[dim]['clamp']
    if pseudosdf_interior is None:
        pseudosdf_interior = default_params[dim]['pseudosdf_interior']

    if state.h is None:
        state.h = default_params[dim]['h']
    if state.U is None:
        assert state.sdf is not None, "If you do not provide U, you must provide an sdf function to sample."
        state.U = _sample_sdf(state.sdf, state.V, state.F)
    if state.S is None:
        assert state.sdf is not None, "If you do not provide S, you must provide an sdf function to sample."
        state.S = state.sdf(state.U)
    if state.min_h is None:
        # use a kdtree and get the average distance between two samples
        # import cKDTree
        tree = cKDTree(state.U)
        dists, _ = tree.query(state.U, k=2)
        state.min_h = 2.*np.mean(dists[:,1])
        state.min_h = np.clip(state.min_h, 0.001, 0.1)

    nu_0 = state.U.shape[0]
    nu_max = 2*nu_0

    if state.rng is None:
        state.rng = np.random.default_rng(68)
    if state.its is None:
        state.its = 0
    if state.best_performance is None:
        state.best_performance = np.inf
    if state.convergence_counter is None:
        state.convergence_counter = 0
    if state.best_avg_error is None:
        state.best_avg_error = np.inf
    # if state.use_features is None:
    #     state.use_features = False
    if state.V_last_converged is None:
        state.V_last_converged = state.V.copy()
    if state.F_last_converged is None:
        state.F_last_converged = state.F.copy()
    if state.resample_counter is None:
        state.resample_counter = 0
    if state.U_batch is None:
        state.U_batch = state.U.copy()
    if state.S_batch is None:
        state.S_batch = state.S.copy()

    remeshing = True

    #Do we prematurely abort? Increment iteration.
    if state.its>=max_iter:
        return True

    #Algorithm
    if batch_size>0 and batch_size<state.U.shape[0]:
        inds = state.rng.choice(state.U.shape[0], batch_size, replace=False)
        state.U_batch = state.U[inds,:]
        state.S_batch = state.S[inds]
        # include all inside points
        state.U_batch = np.concatenate((state.U_batch, state.U[state.S<=0,:]), axis=0)
        state.S_batch = np.concatenate((state.S_batch, state.S[state.S<=0]), axis=0)
    d2, I, b = squared_distance(state.U_batch, state.V, state.F,
        use_cpp=True, use_aabb=True)
    d = np.sqrt(d2)
    g = np.abs(state.S_batch)-d

    #closest point on edge to u
    pe = np.sum(state.V[state.F[I,:],:]*b[...,None], axis=1)
    pemU = pe-state.U_batch
    if inside_outside_test:
        s = -np.sign(np.sum(pemU*_normals(state.V,state.F)[I,:], axis=-1))
        hit_sides = np.any(b<1e-3, axis=-1) #too close to vertex
        s[hit_sides] = np.sign(state.S_batch)[hit_sides]
    else:
        s = np.sign(state.S_batch)
    #closest signed point on sphere to pe
    ps = state.U_batch+pemU*((s*state.S_batch/np.maximum(0.5*tol,d))[:,None])
    valid_U = np.abs(g) < tol
    invalid_U = np.logical_not(valid_U)
    F_invalid = np.unique(I[invalid_U])
    if feature_detection=='aggressive':
        V_invalid = np.unique(state.F[F_invalid,:].ravel())
        feature_vertices = np.setdiff1d(np.arange(state.V.shape[0]), V_invalid)
    else:
        F_valid = np.setdiff1d(np.arange(state.F.shape[0]), F_invalid)
        feature_vertices = np.unique(state.F[F_valid,:])

    #Build matrices
    wu = np.ones(state.U_batch.shape[0])
    clamped_g = np.where((np.abs(state.S_batch)==clamp)*(g<0.))
    if pseudosdf_interior:
        clamped_g = np.where((state.S_batch>0)*(g<0.))
    wu[clamped_g] = 0.0
    A = sp.sparse.csc_matrix(((wu[:,None]*b).ravel(),
        (np.tile(np.arange(state.U_batch.shape[0])[:,None],
            (1,state.F.shape[1])).ravel(),
            state.F[I,:].ravel())),
        (state.U_batch.shape[0],state.V.shape[0]))
    c = wu[:,None]*ps
    M = _massmatrix(state.V, state.F)
    rho = 1.0*np.ones(state.U_batch.shape[0]) / A.shape[0]
    R = sp.sparse.spdiags([rho], 0, rho.shape[0], rho.shape[0])
    
    #Compute timestep t
    if linesearch:
        # Backtracking linesearch
        # Nocedal & Wright Algorithm 3.1
        n_c = 0.01
        n_p = -A.transpose()*R*(A*state.V-c)
        t = -(np.sum((A*state.V-c)*(R*(A*n_p))) + n_c*np.sum(n_p**2)) \
        / np.sum((A*n_p)*(R*(A*n_p)))
        t = np.nan_to_num(t, nan=0., posinf=0., neginf=0.)
        t = min(max_t, max(t,min_t))
    else:
        t = dt

    #Minimize
    Q = M + t*(A.transpose()*R*A)
    b = M*state.V + t*A.transpose()*(R*c)
    if fix_boundary:
        bd = boundary_vertices(state.F)
        precomp = fixed_dof_solve_precompute(Q, k=bd)
        state.V = precomp.solve(b=b, y=state.V[bd,:])
    else:
        state.V = sp.sparse.linalg.spsolve(Q,b)

    # catching flow singularities so we fail gracefully

    there_are_non_manifold_edges = False
    if dim==3:
        there_are_non_manifold_edges = len(non_manifold_edges(state.F))>0
    elif dim==2:
        he_nm = np.sort(state.F, axis=1)
        # print(he)
        he_u_nm = np.unique(he_nm, axis=0, return_counts=True)
        # print(he)
        ne_nm = he_u_nm[0][he_u_nm[1]>2]
        there_are_non_manifold_edges = len(ne_nm)>0
    
    if np.any((np.isnan(state.V))) or np.any(doublearea(state.V, state.F)==0) or there_are_non_manifold_edges : 
        
        if verbose:
            print("we found a flow singularity. Returning the last converged solution.")
        state.V = state.V_last_converged.copy()
        state.F = state.F_last_converged.copy()
        return True

    #Convergence determination
    state.avg_error = np.linalg.norm(A*state.V-c) / A.shape[0]
    if verbose:
        print("Iteration:", state.its, "Counter:",state.convergence_counter, "h:",state.h, "Average error:", state.avg_error, "Best avg error:",state.best_avg_error, "Max error:", np.max(np.linalg.norm(A*state.V-c,axis=1)))
    if state.avg_error+1e-3*tol >= state.best_avg_error:
        state.convergence_counter = state.convergence_counter + 1
    else:
        state.convergence_counter = 0
        state.best_avg_error = state.avg_error
    if state.convergence_counter > 10:
        # if h==min_h:
        #     remeshing = False
        if state.h>state.min_h:
            state.V_last_converged = state.V.copy()
            state.F_last_converged = state.F.copy()
            state.best_avg_error = np.inf
            state.convergence_counter = 0
        state.h = np.maximum(state.h/2,state.min_h)
    if state.convergence_counter > 100 or F_invalid.shape[0] == 0:
        if state.resample_counter<resample:
            state.U = _sample_sdf(state.sdf, state.V, state.F, state.U,
                new_n=resample_samples,
                trial_n=int(50*resample_samples), max_n=nu_max,
                remove_samples=True, keep_these_samples=np.arange(nu_0),
                rng=state.rng)
            state.S = state.sdf(state.U)
            state.U_batch = state.U.copy()
            state.S_batch = state.S.copy()
            state.resample_counter += 1
            state.best_performance = np.inf
            state.convergence_counter = 0
            state.best_avg_error = np.inf
            if verbose:
                print(f"Resampled, I now have {state.U.shape[0]} sample points.")
        else:
            # We have converged.
            return True

    #Remeshing
    if remeshing:
        if (output_sensitive and F_invalid.shape[0] > 0):
            # we find the invalid faces
            F_invalid = np.unique(I[invalid_U])
            # We compute the face adjacency matrix
            TT = _face_adjacency(state.F)
            # We find the set of invalid faces and their neighbors
            F_invalid_neighbors = np.unique(TT[F_invalid,:].ravel())
            # also add the invalid faces
            # F_invalid_neighbors = np.unique(np.hstack((F_invalid_neighbors,F_invalid)))
            F_invalid_neighbors = _one_ring(state.F,F_invalid)
            # do another round of one ring
            F_invalid_neighbors = _one_ring(state.F,F_invalid_neighbors)
            # We find the set of invalid vertices
            # F_active = F[F_invalid_neighbors,:]
            state.V_active, state.F_active, _, _ = remove_unreferenced(state.V,
                state.F[F_invalid_neighbors,:], return_maps=True)

            if (state.F_active.shape[0] < state.F.shape[0] and
                state.F_active.shape[0] > 0):
                # set of inactive faces
                state.F_inactive = np.setdiff1d(
                    np.arange(state.F.shape[0]), F_invalid_neighbors)
                # set of inactive vertices
                state.V_inactive, state.F_inactive, _, _ = remove_unreferenced(
                    state.V, state.F[state.F_inactive,:],return_maps=True)
                # Remesh only the active part
                state.V_active, state.F_active = _remesh(
                    state.V_active, state.F_active, i=remesh_iterations,
                    h=state.h, project=True)
                # We merge the active and inactive parts
                state.V, state.F = _merge_meshes(state.V_active, state.F_active, state.V_inactive, state.F_inactive)
                
            else:
                state.V, state.F = _remesh(state.V, state.F,
                    i=remesh_iterations, h=state.h, project=True)
        else:
            state.V, state.F = _remesh(state.V, state.F,
                i=remesh_iterations, h=state.h, project = True,
                feature=feature_vertices)
            state.V_active = None
            state.F_active = None
            state.V_inactive = None
            state.F_inactive = None
    else:
        state.V_active = None
        state.F_active = None
        state.V_inactive = None
        state.F_inactive = None

    #Have we converged?
    state.its = state.its+1
    if state.its>=max_iter:
        return True
    else:
        return False


def _face_adjacency(F):
    dim = F.shape[1]
    if dim==2:
        n = np.max(F)+1
        v_to_f = -np.ones((n,2), dtype=int)
        v_to_f[F[:,0],0] = np.arange(F.shape[0])
        v_to_f[F[:,1],1] = np.arange(F.shape[0])
        TT = np.stack((v_to_f[F[:,0],1], v_to_f[F[:,1],0]), axis=-1)
        return TT
    elif dim==3:
        TT,_ = triangle_triangle_adjacency(F)
        return TT


def _normals(V,F,unit_norm=False):
    dim = F.shape[1]
    if dim==2:
        e = V[F[:,1],:] - V[F[:,0],:]
        if unit_norm:
            e /= np.linalg.norm(e, axis=-1)[:,None]
        return e @ np.array([[0., -1.], [1., 0.]])
    elif dim==3:
        return per_face_normals(V,F,unit_norm=unit_norm)

    
def _massmatrix(V,F):
    dim = F.shape[1]
    if dim==3:
        l_sq = halfedge_lengths_squared(V,F)
        l_sq = np.maximum(l_sq, 100.*np.finfo(V.dtype).eps)
        M = massmatrix_intrinsic(l_sq,F,n=V.shape[0]) 
    elif dim==2:
        edge_lengths = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        edge_lengths = np.maximum(edge_lengths, np.sqrt(100.*np.finfo(V.dtype).eps))
        vals = np.concatenate((edge_lengths,edge_lengths))/2.
        I = np.concatenate((F[:,0],F[:,1]))
        M = sp.sparse.csr_matrix((vals,(I,I)),shape=(V.shape[0],V.shape[0]))
    return M


def _one_ring(F,active_faces):
    # Vectorized
    active_verts = F[active_faces,:].ravel()
    one_ring_active = np.nonzero(np.isin(F,active_verts).any(axis=-1))[0]

    # # Step 1: Construct adjacency list for each vertex
    # n = np.max(F)+1
    # adjacency_list = {i: [] for i in range(n)}
    # for i, face in enumerate(F):
    #     for vertex in face:
    #         adjacency_list[vertex].append(i)

    # # Step 2: Iterate over active faces and their adjacent faces
    # one_ring_active = set(active_faces)  # start with active faces
    # for face in active_faces:
    #     for vertex in F[face]:
    #         one_ring_active.update(adjacency_list[vertex])  # add adjacent faces

    # one_ring_active = np.array(list(one_ring_active))

    return one_ring_active


def _remesh(V, F, i=10, h=None, project=True, feature=np.array([], dtype=int)):
    dim = F.shape[1]
    if dim==2:
        V,F = _remesh_line(V, F,
            i=i, h=h,
            project=project, feature=feature)
    elif dim==3:
        if V.shape[0]==len(feature):
            # then the call to remesh_botsch is a no-op, because our flow has converged (active region is empty)
            V,F = V.copy(), F.copy()
        else:
            V,F = remesh_botsch(V, F,
                i=i, h=h,
                project=project, feature=feature)
    assert np.isfinite(V).all()
    return V,F


def _line_bdry(F):
    return np.unique(np.concatenate((
        np.setdiff1d(F[:,0],F[:,1]),
        np.setdiff1d(F[:,1],F[:,0])
        )))


def _remesh_line(V, F, i=10, h=None, project=True, feature=np.array([], dtype=int)):
    high = 1.4*h
    low = 0.7*h
    V0,F0 = V.copy(), F.copy()
    feature = np.unique(np.concatenate((feature, _line_bdry(F))))
    for its in range(i):
        V,F,feature = _split_line(V, F, feature, high, low)
        V,F,feature = _collapse_line(V, F, feature, high, low)
        if not project:
            V0,F0 = V.copy(), F.copy()
        V,F = _relaxation_line(V, F, feature, V0, F0)
    return V,F


def _split_line(V, F, feature, high, low):
    n = V.shape[0]
    # is_feature = np.full(V.shape[0], False, dtype=bool)
    # is_feature[feature] = True
    # can_split = np.logical_and(np.logical_not(is_feature[F[:,0]]),
    #     np.logical_not(is_feature[F[:,1]]))
    can_split = np.full(F.shape[0], True, dtype=bool)

    to_split = np.nonzero(np.logical_and(can_split,
        np.linalg.norm(V[F[:,1]]-V[F[:,0]],axis=-1)>high))[0]

    V = np.concatenate((V, 0.5*(V[F[to_split,0],:]+V[F[to_split,1],:])), axis=0)
    new_verts = np.arange(n,n+to_split.size)
    F = np.concatenate(
        (F, np.stack((new_verts, F[to_split,1]), axis=-1)),
    axis=0)
    F[to_split,1] = new_verts

    return V,F,feature


def _collapse_line(V, F, feature, high, low):
    is_feature = np.full(V.shape[0], False, dtype=bool)
    is_feature[feature] = True

    while True:
        l = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:], axis=-1)
        a = np.argsort(l)
        collapse = -1
        for i in range(a.size):
            if l[a[i]]<low:
                if (not is_feature[F[a[i],0]] or not is_feature[F[a[i],1]]):
                    collapse = a[i]
                    break
            else:
                break
        if collapse<0:
            break
        if not is_feature[F[collapse,0]] and not is_feature[F[collapse,1]]:
            remove = F[collapse,1]
            keep = F[collapse,0]
            mid = 0.5*(V[remove,:]+V[keep,:])
            V[keep,:] = mid
        elif is_feature[F[collapse,0]]:
            remove = F[collapse,1]
            keep = F[collapse,0]
        elif is_feature[F[collapse,1]]:
            remove = F[collapse,0]
            keep = F[collapse,1]
        assert remove not in feature
        assert not is_feature[remove]
        F[F==remove] = keep

        F = np.delete(F, collapse, axis=0)
        V,F,I,J = remove_unreferenced(V, F, return_maps=True)
        feature = I[feature]
        is_feature = np.full(V.shape[0], False, dtype=bool)
        is_feature[feature] = True

    return V,F,feature


def _relaxation_line(V, F, feature, V0, F0):
    is_feature = np.full(V.shape[0], False, dtype=bool)
    is_feature[feature] = True

    l = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:], axis=-1)
    face_N = _normals(V,F,unit_norm=True)
    N = np.zeros((V.shape[0],2))
    N[F[:,0],:] += face_N*l[:,None]
    N[F[:,1],:] += face_N*l[:,None]
    N /= np.linalg.norm(N, axis=-1)[:,None]

    adj = -np.ones((V.shape[0],2), dtype=int)
    adj[F[:,0],0] = F[:,1]
    adj[F[:,1],1] = F[:,0]

    for i in range(V.shape[0]):
        if is_feature[i]:
            continue
        if not np.isfinite(N[i,:]).all():
            continue
        if not (adj[i,:]>=0).all():
            continue
        q = 0.5 * (V[adj[i,0],:]+V[adj[i,1],:])
        NN = np.identity(2)-N[i,:][:,None]*N[i,:][None,:]
        V[i,:] -= NN@(V[i,:]-q)

    _,I,b = squared_distance(V,V0,F0,use_cpp=True,use_aabb=True)
    V = np.sum(V0[F0[I,:],:]*b[...,None], axis=1)

    return V,F


#Bounds for a set of points
def _bounds(V, tol=0.):
    lb = np.min(V, axis=0)
    ub = np.max(V, axis=0)
    lb -= (ub-lb)*0.5 - tol
    ub += (ub-lb)*0.5 + tol
    return lb,ub


#void distance function for a given SDF S at points U, evaluated at x
def _vdf(x, U, S):
    vf = S[None,:]**2 - np.sum((x[:,None,:]-U[None,:,:])**2, axis=-1)
    v = np.max(vf, axis=1)
    v = np.minimum(v,0.)
    return v


def _sample_sdf(sdf,
    V, F,
    U0 = None, #Initial evaluation points
    new_n = 20, #How many new samples to add each step
    trial_n = 1000, #How many new samples to try before deciding which to add
    max_n = 100000, #Maximum size of U
    remove_samples = False, #Whether to remove some samples when close to max_n
    keep_these_samples = None, #Samples protected from removal
    rng = np.random.default_rng(), #Which rng to use
    tol = 1e-3):

    dim = V.shape[1]
    assert dim==F.shape[1]
    assert remove_samples or max_n>=U0.shape[0], "Either allow removal or start with not too many samples."

    if U0 is None:
        #Randomly sample twice as many points as initial mesh vertices.
        lb,ub = _bounds(V, tol)
        n = min(max_n, V.shape[0])
        dim = V.shape[1]
        U0 = rng.uniform(lb, ub, size=(n,dim))
    S0 = sdf(U0)

    # Random points on all faces
    P,I,_ = random_points_on_mesh(V, F, trial_n, rng=rng, return_indices=True)
    d = 0.05 * rng.normal(scale=np.max(np.max(V,axis=0)-np.min(V,axis=0)), size=P.shape[0])
    P += d[:,None] * _normals(V, F, unit_norm=True)[I,:]
    # Remove all points in P that are not worst points on edge.
    worst = {}
    for i in range(P.shape[0]):
        if (I[i] not in worst) or (_vdf(P[i,:][None,:],U0,S0)
            <_vdf(P[worst[I[i]],:][None,:],U0,S0)):
            worst[I[i]] = i
    P = np.array([P[i,:] for _,i in worst.items()])
    # Get new_n worst points
    I = np.argsort(_vdf(P, U0, S0))
    U = np.concatenate((U0, P[I[:new_n]]), axis=0)

    if U.shape[0] <= max_n:
        return U
    else:
        if remove_samples:
            Sa = np.abs(sdf(U))
            if keep_these_samples is not None:
                Sa[keep_these_samples] = 0.
            I = np.argsort(Sa)
            return U[I[:max_n],:]
        else:
            return U[:max_n,:]

def _merge_meshes(V_active, F_active, V_inactive, F_inactive):
    """ Combines the active and inactive mesh while making sure to not merge any *interior* active vertex with a boundary active vertex, avoiding the creation of a non-manifold mesh as shown in gh issue 100 """
    dim = V_active.shape[1]

    if dim==3:
        bd_active = boundary_vertices(F_active) # boundary
        interior_active = np.setdiff1d(np.arange(V_active.shape[0]), bd_active)
        V_for_zipping = np.vstack((V_inactive, V_active[bd_active,:]))
        
        # now we will merge the boundary vertices of the active mesh with the inactive mesh
        _, zipped_indices, zipped_indices_inverse = np.unique(np.round(V_for_zipping/np.sqrt(np.finfo(V_active.dtype).eps)),return_index=True,return_inverse=True,axis=0)
        # add the interior vertices of the active mesh manually into the index list, making sure they are not merged with anything else:
        unique_num = zipped_indices.shape[0] # number of unique vertices
        zipped_indices = np.concatenate((zipped_indices, interior_active + V_inactive.shape[0]))
        # inverse map update
        zipped_indices_inverse = np.concatenate((zipped_indices_inverse, np.arange(interior_active.shape[0]) + unique_num))
        # now use the index maps to get the final mesh
        V = np.vstack((V_inactive, V_active))
        F_for_zipping = np.vstack((F_inactive, F_active + V_inactive.shape[0]))
        F = zipped_indices_inverse[F_for_zipping]
        V = V[zipped_indices,:]
    elif dim==2:
        V = np.vstack((V_inactive, V_active))
        F = np.vstack((F_inactive, F_active + V_inactive.shape[0]))
        V, _, _, F = remove_duplicate_vertices(V, faces=F, epsilon=np.sqrt(np.finfo(V.dtype).eps))
    return V,F