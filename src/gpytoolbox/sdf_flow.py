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


def sdf_flow(U, sdf, V, F, S=None,
    return_U=False,
    verbose=False,
    visualize=False, #TODO: remove
    callback=None, #TODO: remove
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
    clamp=None, sv=None):

    state = SdfFlowState(V=V, F=F, sdf=sdf, U=U, S=S,
        h=h, min_h=min_h)
    converged = False

    # Little hack to pass max_iter, which is not keps as state.
    # Set to the same default as in sdf_flow_iteration.
    dim = V.shape[1]
    pass_max_iter = (10000 if dim==2 else 20000
        ) if max_iter is None else max_iter

    # TODO: Replace this function with a simple while loop that breaks if converged.
    def run_flow_iteration():
        nonlocal state, converged
        if not converged:
            converged = sdf_flow_iteration(state,
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
                clamp=clamp, sv=sv)

            # TODO: remove callback code
            if callback is not None:
                callback({'V':state.V, 'F':state.F,
                    'V_active':state.V_active,
                    'F_active':state.F_active,
                    'V_inactive':state.V_inactive,
                    'F_inactive':state.F_inactive,
                    'its':state.its,
                    'max_iter':pass_max_iter,
                    'U':state.U, 'S':state.S,
                    'converged':converged,
                    'resample_counter':state.resample_counter})

            # TODO: remove visualization code
            if visualize:
                if dim==2:
                    def plot_edges(vv,ee,plt_str):
                        ax = plt.gca()
                        for i in range(ee.shape[0]):
                            ax.plot([vv[ee[i,0],0],vv[ee[i,1],0]],
                                     [vv[ee[i,0],1],vv[ee[i,1],1]],
                                     plt_str,alpha=1)
                    def plot_spheres(vv,sdf):
                        ax = plt.gca()
                        f = sdf(vv)
                        for i in range(vv.shape[0]):
                            c = 'r' if f[i]>=0 else 'b'
                            ax.add_patch(plt.Circle(vv[i,:], f[i], color=c, fill=False,alpha=0.1))
                    plt.cla()
                    plot_spheres(U,sdf)
                    visualize_full = True
                    if state.V_active is not None and state.F_active is not None:
                        visualize_full = False
                        plot_edges(state.V_active,state.F_active,'b-')
                        plt.plot(state.V_active[:,0],state.V_active[:,1],'b.')
                    if state.V_inactive is not None and state.F_inactive is not None:
                        visualize_full = False
                        plot_edges(state.V_inactive,state.F_inactive,'y-')
                        plt.plot(state.V_inactive[:,0],state.V_inactive[:,1],'y.')
                    if visualize_full and state.V is not None and state.F is not None:
                        plot_edges(state.V,state.F,'b-')
                        plt.plot(state.V[:,0],state.V[:,1],'b.')
                    plt.draw()
                    plt.pause(0.01)
                elif dim==3:
                    # if stopped:
                    #     # This mess is so that we can render something from polyscope in the same script, otherwise this callback will keep executing and deleting everything you plot.
                    #     if active_ps is not None:
                    #         active_ps.remove()
                    #         active_ps = None
                    #     if inactive_ps is not None:
                    #         inactive_ps.remove()
                    #         inactive_ps = None
                    #     if full_ps is not None:
                    #         full_ps.remove()
                    #         full_ps = None
                    #     # polyscope.remove_all_structures()
                    # else:
                        # cloud_U = polyscope.register_point_cloud("SDF evaluation points", U)
                        # cloud_U.add_scalar_quantity("How unhappy?", np.abs(g), enabled=True)
                        visualize_full = True
                        if state.V_active is not None and state.F_active is not None:
                            visualize_full = False
                            active_ps = polyscope.register_surface_mesh("active", state.V_active, state.F_active)
                        if state.V_inactive is not None and state.F_inactive is not None:
                            visualize_full = False
                            inactive_ps = polyscope.register_surface_mesh("inactive", state.V_inactive, state.F_inactive)
                        if visualize_full and V is not None and F is not None:
                            full_ps = polyscope.register_surface_mesh("full", state.V, state.F)

    # TODO: remove visualization code
    if visualize and dim==3:
        import polyscope
        polyscope.init()
        def polyscope_callback():
            run_flow_iteration()
        polyscope.set_user_callback(polyscope_callback)
        polyscope.show()
    else:
        if visualize:
            import matplotlib.pyplot as plt
        while state.its is None or (state.its<pass_max_iter and (not converged)):
            run_flow_iteration()

    if return_U:
        return state.V, state.F, state.U
    else:
        return state.V, state.F


class SdfFlowState:

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
        # use_features=None,
        resample_counter=None,
        full_ps=None, active_ps=None, inactive_ps=None,
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
        # self.use_features = use_features
        self.resample_counter = resample_counter
        self.full_ps = full_ps
        self.active_ps = active_ps
        self.inactive_ps = inactive_ps
        self.V_last_converged = V_last_converged
        self.F_last_converged = F_last_converged
        self.U_batch = U_batch
        self.S_batch = S_batch


#Returns true if algorithm has converged.
def sdf_flow_iteration(state,
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
    clamp=None, sv=None,
    verbose=False):


    assert isinstance(state, SdfFlowState), "State must be a SdfFlowState"
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
        'clamp':np.Inf, 'sv':False},
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
        'clamp':np.Inf, 'sv':False}
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
    if sv is None:
        sv = default_params[dim]['sv']

    if state.h is None:
        state.h = default_params[dim]['h']
    if state.U is None:
        assert state.sdf is not None, "If you do not provide U, you must provide an sdf function to sample."
        state.U = sample_sdf(state.sdf, state.V, state.F)
    if state.S is None:
        assert state.sdf is not None, "If you do not provide U, you must provide an sdf function to sample."
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
        state.best_performance = np.Inf
    if state.convergence_counter is None:
        state.convergence_counter = 0
    if state.best_avg_error is None:
        state.best_avg_error = np.Inf
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
    state.its = state.its+1 #Only here for compatibility, move to end after.

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
        s = -np.sign(np.sum(pemU*normals(state.V,state.F)[I,:], axis=-1))
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
    if sv:
        clamped_g = np.where((state.S_batch>0)*(g<0.))
    wu[clamped_g] = 0.0
    A = sp.sparse.csc_matrix(((wu[:,None]*b).ravel(),
        (np.tile(np.arange(state.U_batch.shape[0])[:,None],
            (1,state.F.shape[1])).ravel(),
            state.F[I,:].ravel())),
        (state.U_batch.shape[0],state.V.shape[0]))
    c = wu[:,None]*ps
    M = massmatrix(state.V, state.F)
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
        state.V = precomp.solve(b=b, y=V[bd,:])
    else:
        state.V = sp.sparse.linalg.spsolve(Q,b)

    # catching flow singularities so we fail gracefully
    if np.any((np.isnan(state.V))):
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
            state.best_avg_error = np.Inf
            state.convergence_counter = 0
        state.h = np.maximum(state.h/2,state.min_h)
        # state.use_features = True
    converged = False
    if state.convergence_counter > 100 or F_invalid.shape[0] == 0:
        if state.resample_counter<resample:
            state.U = sample_sdf(state.sdf, state.V, state.F, state.U,
                new_n=resample_samples,
                trial_n=int(50*resample_samples), max_n=nu_max,
                remove_samples=True, keep_these_samples=np.arange(nu_0),
                rng=state.rng)
            state.S = state.sdf(state.U)
            state.U_batch = state.U.copy()
            state.S_batch = state.S.copy()
            state.resample_counter += 1
            state.best_performance = np.Inf
            state.convergence_counter = 0
            state.best_avg_error = np.Inf
            # state.min_h = max(0.001, 0.8*state.min_h)
            if verbose:
                print(f"Resampled, I now have {state.U.shape[0]} sample points.")
        else:
            # We have converged.
            return True

    #Remeshing
    if remeshing:
        # if (not state.use_features):
        #     feature_vertices = np.array([],dtype=np.int32)

        if (output_sensitive and F_invalid.shape[0] > 0):
            # we find the invalid faces
            F_invalid = np.unique(I[invalid_U])
            # We compute the face adjacency matrix
            TT = face_adjacency(state.F)
            # We find the set of invalid faces and their neighbors
            F_invalid_neighbors = np.unique(TT[F_invalid,:].ravel())
            # also add the invalid faces
            # F_invalid_neighbors = np.unique(np.hstack((F_invalid_neighbors,F_invalid)))
            F_invalid_neighbors = one_ring(state.F,F_invalid)
            # do another round of one ring
            F_invalid_neighbors = one_ring(state.F,F_invalid_neighbors)
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
                
                state.V_active, state.F_active = remesh(
                    state.V_active, state.F_active, i=remesh_iterations,
                    h=state.h, project=True)
                # We merge the active and inactive parts
                state.V = np.vstack((state.V_active, state.V_inactive))
                state.F = np.vstack((state.F_active,
                    state.F_inactive + state.V_active.shape[0]))
                # We remove the duplicate vertices
                state.V,_,_,state.F = remove_duplicate_vertices(
                    state.V,faces=state.F,
                    epsilon=np.sqrt(np.finfo(state.V.dtype).eps))
            else:
                state.V, state.F = remesh(state.V, state.F,
                    i=remesh_iterations, h=state.h, project=True)
        else:
            state.V, state.F = remesh(state.V, state.F,
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
    if state.its>=max_iter:
        return True
    else:
        return False


def face_adjacency(F):
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


def laplacian(v,f):
    dim = v.shape[1]
    if dim==3:
        L = cotangent_laplacian(v,f)
    elif dim==2:
        G = grad(v,f)
        A = 0.5*doublearea(v,f)
        MA = sp.sparse.spdiags(np.concatenate((A,A)), 0, G.shape[0], G.shape[0])
        L = G.transpose() * MA * G
    L.data[np.logical_not(np.isfinite(L.data))] = 0.
    return L


def normals(V,F,unit_norm=False):
    dim = F.shape[1]
    if dim==2:
        e = V[F[:,1],:] - V[F[:,0],:]
        if unit_norm:
            e /= np.linalg.norm(e, axis=-1)[:,None]
        return e @ np.array([[0., -1.], [1., 0.]])
    elif dim==3:
        return per_face_normals(V,F,unit_norm=unit_norm)

  
def per_vertex_normals(V,F):
    N = normals(V,F,unit_norm=True)
    N = np.nan_to_num(N, nan=0., posinf=0., neginf=0.)
    Fr = F.ravel(order='F')
    dim = F.shape[1]
    if dim==2:
        Nv = np.stack((
            np.bincount(Fr, weights=np.concatenate((N[:,0],N[:,0]))),
            np.bincount(Fr, weights=np.concatenate((N[:,1],N[:,1])))
            ), axis=-1)
    elif dim==3:
        α = tip_angles(V,F)
        αr = α.ravel(order='F')
        Nv = np.stack((
            np.bincount(Fr, weights=αr*np.concatenate((N[:,0],N[:,0],N[:,0]))),
            np.bincount(Fr, weights=αr*np.concatenate((N[:,1],N[:,1],N[:,1]))),
            np.bincount(Fr, weights=αr*np.concatenate((N[:,2],N[:,2],N[:,2])))
            ), axis=-1)
    Nv /= np.linalg.norm(Nv, axis=-1)[:,None]
    Nv = np.nan_to_num(Nv, nan=0., posinf=0., neginf=0.)
    return Nv


def barycentric_normals(I,b,V,F,unit_norm=False):
    Nv = per_vertex_normals(V,F)
    N = np.sum(Nv[F[I,:],:]*b[...,None], axis=1)
    if unit_norm:
        N /= np.linalg.norm(N, axis=-1)[:,None]
    N = np.nan_to_num(N, nan=0., posinf=0., neginf=0.)
    return N


def processed_normals(I,b,V,F,unit_norm=False):
    Nb = barycentric_normals(I,b,V,F,unit_norm=unit_norm)
    return Nb

    
def massmatrix(V,F):
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


def one_ring(F,active_faces):
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


def remesh(V, F, i=10, h=None, project=True, feature=np.array([], dtype=int)):
    dim = F.shape[1]
    if dim==2:
        V,F = remesh_line(V, F,
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


def line_bdry(F):
    return np.unique(np.concatenate((
        np.setdiff1d(F[:,0],F[:,1]),
        np.setdiff1d(F[:,1],F[:,0])
        )))


def remesh_line(V, F, i=10, h=None, project=True, feature=np.array([], dtype=int)):
    high = 1.4*h
    low = 0.7*h
    V0,F0 = V.copy(), F.copy()
    feature = np.unique(np.concatenate((feature, line_bdry(F))))
    for its in range(i):
        V,F,feature = split_line(V, F, feature, high, low)
        V,F,feature = collapse_line(V, F, feature, high, low)
        if not project:
            V0,F0 = V.copy(), F.copy()
        V,F = relaxation_line(V, F, feature, V0, F0)
    return V,F


def split_line(V, F, feature, high, low):
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


def collapse_line(V, F, feature, high, low):
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


def relaxation_line(V, F, feature, V0, F0):
    is_feature = np.full(V.shape[0], False, dtype=bool)
    is_feature[feature] = True

    l = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:], axis=-1)
    face_N = normals(V,F,unit_norm=True)
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
def bounds(V, tol=0.):
    lb = np.min(V, axis=0)
    ub = np.max(V, axis=0)
    lb -= (ub-lb)*0.5 - tol
    ub += (ub-lb)*0.5 + tol
    return lb,ub


#void distance function for a given SDF S at points U, evaluated at x
def vdf(x, U, S):
    vf = S[None,:]**2 - np.sum((x[:,None,:]-U[None,:,:])**2, axis=-1)
    v = np.max(vf, axis=1)
    v = np.minimum(v,0.)
    return v


def sample_sdf(sdf,
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
        lb,ub = bounds(V, tol)
        n = min(max_n, V.shape[0])
        dim = V.shape[1]
        U0 = rng.uniform(lb, ub, size=(n,dim))
    S0 = sdf(U0)

    # Random points on all faces
    P,I,_ = random_points_on_mesh(V, F, trial_n, rng=rng, return_indices=True)
    d = 0.05 * rng.normal(scale=np.max(np.max(V,axis=0)-np.min(V,axis=0)), size=P.shape[0])
    P += d[:,None] * normals(V, F, unit_norm=True)[I,:]
    # Remove all points in P that are not worst points on edge.
    worst = {}
    for i in range(P.shape[0]):
        if (I[i] not in worst) or (vdf(P[i,:][None,:],U0,S0)
            <vdf(P[worst[I[i]],:][None,:],U0,S0)):
            worst[I[i]] = i
    P = np.array([P[i,:] for _,i in worst.items()])
    # Get new_n worst points
    I = np.argsort(vdf(P, U0, S0))
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

