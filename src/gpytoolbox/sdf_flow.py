# Here I import only the functions I need for these functions
import numpy as np
import scipy as sp
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox/src')))
import gpytoolbox as gpy
import matplotlib.pyplot as plt
import polyscope
from gpytoolbox.copyleft import mesh_boolean
from scipy.sparse import coo_matrix, block_diag, diags
from .laplacian import laplacian
from .massmatrix import massmatrix
from .normals import normals, processed_normals
from .sample_sdf import sample_sdf
from .one_ring import one_ring
from .remesh import remesh
from .face_adjacency import face_adjacency
from scipy.spatial import cKDTree


def sdf_flow(U, sdf, V, F, S=None,
    max_iter=None, tol=None, h=None, min_h=None,
    linesearch=None, max_t=None,
    dt=None,
    inside_outside_test=None,
    resample=None,
    resample_samples=None,
    return_U=None,
    feature_detection=None,
    output_sensitive=None,
    remesh_iterations=None,
    verbose=None,
    callback=None,
    visualize=None,
    batch_size=None,
    fix_boundary=False,
    clamp=np.Inf, sv=False):

    dim = U.shape[1]
    assert dim==V.shape[1]
    assert dim==F.shape[1]

    #Assign different 2D and 3D default parameters
    default_params = {
        #For each dimension
        2: {'max_iter':10000, 'tol':5e-3, 'h':0.1, 'min_h':None,
        'linesearch':True, 'max_t':50.,
        'dt':10.,
        'inside_outside_test':True,
        'resample':0, 'resample_samples':2*int(np.ceil(np.sqrt(U.shape[0]))),
        'return_U':False,
        'feature_detection':'aggressive', 'output_sensitive':True,
        'remesh_iterations':1, 'verbose':True, 'callback':None,
        'visualize':False,
        'batch_size':20000},
        3: {'max_iter':20000, 'tol':1e-2, 'h':0.2, 'min_h':None,
        'linesearch':True, 'max_t':50.,
        'dt':10.,
        'inside_outside_test':True,
        'resample':0, 'resample_samples':2*int(np.ceil(np.cbrt(U.shape[0]))),
        'return_U':False,
        'feature_detection':'aggressive', 'output_sensitive':True,
        'remesh_iterations':1, 'verbose':True, 'callback':None,
        'visualize':False,
        'batch_size':20000}
    }
    if max_iter is None:
        max_iter = default_params[dim]['max_iter']
    if tol is None:
        tol = default_params[dim]['tol']
    if h is None:
        h = default_params[dim]['h']
    if min_h is None:
        min_h = default_params[dim]['min_h']
    if linesearch is None:
        linesearch = default_params[dim]['linesearch']
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
    if return_U is None:
        return_U = default_params[dim]['return_U']
    if feature_detection is None:
        feature_detection = default_params[dim]['feature_detection']
    if output_sensitive is None:
        output_sensitive = default_params[dim]['output_sensitive']
    if remesh_iterations is None:
        remesh_iterations = default_params[dim]['remesh_iterations']
    if verbose is None:
        verbose = default_params[dim]['verbose']
    if callback is None:
        callback = default_params[dim]['callback']
    if visualize is None:
        visualize = default_params[dim]['visualize']
    if batch_size is None:
        batch_size = default_params[dim]['batch_size']

    if U is None:
        U = sample_sdf(sdf, V, F)

    if S is None:
        S = sdf(U)

    if min_h is None:
        # use a kdtree and get the average distance between two samples
        # import cKDTree
        tree = cKDTree(U)
        dists, _ = tree.query(U, k=2)
        min_h = 2.*np.mean(dists[:,1])
        min_h = np.clip(min_h, 0.001, 0.1)


    nu_0 = U.shape[0]
    nu_max = 2*nu_0
    converged = False
    rng = np.random.default_rng(68)
    its = 0
    best_performance = np.Inf
    convergence_counter = 0
    best_avg_error = np.Inf
    feature = np.array([],dtype=np.int32)
    use_features = False
    remeshing = True
    stopped = False
    V_active, F_active = None, None
    V_inactive, F_inactive = None, None
    V_last_converged, F_last_converged = V.copy(), F.copy()
    resample_counter = 0
    full_ps, active_ps, inactive_ps = None, None, None # for visualization
    U_batch,S_batch = U.copy(),S.copy()
    
    def run_flow_iteration():
        nonlocal V, F, U, S, max_iter, V_active, F_active, V_inactive, F_inactive, tol, h, min_h, its, convergence_counter, best_performance, best_avg_error, feature, use_features, remeshing, feature_detection, converged, stopped, max_t, resample_counter, full_ps, active_ps, inactive_ps, V_last_converged, F_last_converged, U_batch, S_batch

        if (its<max_iter and (not converged)):
            if batch_size>0. and batch_size<U.shape[0]:
                inds = rng.choice(U.shape[0], batch_size, replace=False)
                U_batch = U[inds,:]
                S_batch = S[inds]
                # include all inside points
                U_batch = np.concatenate((U_batch, U[S>0,:]), axis=0)
                S_batch = np.concatenate((S_batch, S[S>0]), axis=0)
            its = its+1
            d2, I, b = gpy.squared_distance(U_batch, V, F, use_cpp=True, use_aabb=True)
            d = np.sqrt(d2)
            g = np.abs(S_batch)-d
            # if g is negative then we are missing the sphere. If S_batch is clamp or more, then we are going to make rho zero
            

            pe = np.sum(V[F[I,:],:]*b[...,None], axis=1) #closest point on edge to u
            pemU = pe-U_batch
            if inside_outside_test:
                # N = processed_normals(I,b,V,F)
                # s = -np.sign(np.sum(pemU*N, axis=-1))
                s = -np.sign(np.sum(pemU*normals(V,F)[I,:], axis=-1))
                s *= -1. if dim==3 else 1. #WHY IS THERE A MINUS HERE in 3D?
                hit_sides = np.any(b<1e-3, axis=-1) #too close to vertex
                s[hit_sides] = np.sign(S_batch)[hit_sides]
            else:
                s = np.sign(S_batch)
            ps = U_batch+pemU*((s*S_batch/np.maximum(0.5*tol,d))[:,None]) #closest point on sphere to pe
            valid_U = np.abs(g) < tol
            invalid_U = np.logical_not(valid_U)
            F_invalid = np.unique(I[invalid_U])
            if feature_detection=='aggressive':
                V_invalid = np.unique(F[F_invalid,:].ravel())
                feature_vertices = np.setdiff1d(np.arange(V.shape[0]), V_invalid)
            else:
                F_valid = np.setdiff1d(np.arange(F.shape[0]), F_invalid)
                feature_vertices = np.unique(F[F_valid,:])

            # Build ADMM matrices
            # ADMM Lagrangian:
            # L = 0.5*||V-V0||_W^2 + rho/2*||A*V-c+y||^2
            wu = np.ones(U_batch.shape[0])
            clamped_g = np.where((np.abs(S_batch)==clamp)*(g<0.))
            if sv:
                clamped_g = np.where((S_batch>0)*(g<0.))
            wu[clamped_g] = 0.0
            # polyscope.register_point_cloud("U_batch", U_batch[clamped_g] )
            # wu[(S_batch<0)*(g<tol)] = 2.0
            A = sp.sparse.csc_matrix(((wu[:,None]*b).ravel(),
                (np.tile(np.arange(U_batch.shape[0])[:,None], (1,F.shape[1])).ravel(),
                    F[I,:].ravel())),
                (U_batch.shape[0],V.shape[0]))
            c = wu[:,None]*ps
            M = massmatrix(V,F)
            rho = 1.0*np.ones(U_batch.shape[0]) / A.shape[0]
            R = sp.sparse.spdiags([rho],0,rho.shape[0],rho.shape[0])
            
            if linesearch:
                # Backtracking linesearch
                # Nocedal & Wright Algorithm 3.1
                n_c = 0.01
                n_p = -A.transpose()*R*(A*V-c)
                t = -(np.sum((A*V-c)*(R*(A*n_p))) + n_c*np.sum(n_p**2)) \
                / np.sum((A*n_p)*(R*(A*n_p)))
                # print(f"computed t: {t}")
                t = np.nan_to_num(t, nan=0., posinf=0., neginf=0.)
                t = min(max_t, max(t,1e-6))
                # print(f"actual t: {t}")
            else:
                t = dt
            
            # Minimize 
            # Q = 0.5*||V-V0||_W^2 + lambda*||A*V-c||_^2
            # make M = I
            # M = 0.001*sp.sparse.eye(V.shape[0])
            # rho[(S_batch>0)*(g>0)] = 50.0*rho[(S_batch>0)*(g>0)]
            # rho[(S_batch>0)*(g<0)] = 0.05*rho[(S_batch>0)*(g<0)]
            # print(np.linalg.norm(A[(S_batch>0)*(g>0)]*V-c[(S_batch>0)*(g>0)]))
            # polyscope.register_point_cloud("full", U_batch[(S_batch>0)*(g>0),:])
            # polyscope.show()
            Q = M + t*(A.transpose()*R*A)
            b = M*V + t*A.transpose()*(R*c)
            # if np.any((np.isnan(V))):
            #     print("NAN BEFORE SOLVE")
            # save mesh before solve
            # gpy.write_mesh(f"before_solve.obj", V, F)
            if fix_boundary:
                bd = gpy.boundary_vertices(F)
                precomp = gpy.fixed_dof_solve_precompute(Q, k=bd)
                # V_new = 0*V
                # for dd in [0,2]:
                #     V_new[:,dd] = precomp.solve(b=b[:,dd], y=V[bd,dd])
                # V_new[:,1] = precomp.solve(b=None,y=None)
                # V = V_new.copy()
                V = precomp.solve(b=b, y=V[bd,:])
                # print("size of bd:", bd.shape[0], "size of V:", V.shape[0])
            else:
                V = sp.sparse.linalg.spsolve(Q,b)
            # gpy.write_mesh(f"after_solve.obj", V, F)
            

            avg_error = np.linalg.norm(A*V-c) / A.shape[0]
            if verbose:
                print("Iteration:", its, "Counter:",convergence_counter, "h:",h, "Average error:", avg_error, "Best avg error:",best_avg_error, "Max error:", np.max(np.linalg.norm(A*V-c,axis=1)))

            # catching flow singularities so we fail gracefully
            if np.any((np.isnan(V))):
                if verbose:
                    print("we found a flow singularity. Returning the last converged solution.")
                V = V_last_converged.copy()
                F = F_last_converged.copy()
                converged = True
                remeshing = False

            if avg_error+1e-3*tol >= best_avg_error:
                convergence_counter = convergence_counter + 1
            else:
                convergence_counter = 0
                best_avg_error = avg_error
            if convergence_counter > 10:
                
                # max_t = max_t/2.0
                # if h==min_h:
                #     remeshing = False
                if h>min_h:
                    V_last_converged = V.copy()
                    F_last_converged = F.copy()
                    best_avg_error = np.Inf
                    convergence_counter = 0
                h = np.maximum(h/2,min_h)
                # convergence_counter = 0
                # best_avg_error = np.Inf
                use_features = True
                # feature = feature_vertices.copy()
            if convergence_counter > 100 or F_invalid.shape[0] == 0:
                if resample_counter<resample:
                    U = sample_sdf(sdf, V, F, U, new_n=resample_samples,
                        trial_n=int(50*resample_samples), max_n=nu_max,
                        remove_samples=True, keep_these_samples=np.arange(nu_0),
                        rng=rng)
                    S = sdf(U)
                    U_batch,S_batch = U.copy(),S.copy()
                    resample_counter += 1
                    best_performance = np.Inf
                    convergence_counter = 0
                    best_avg_error = np.Inf
                    # min_h = max(0.001, 0.8*min_h)
                    if verbose:
                        print(f"Resampled, I now have {U.shape[0]} sample points.")
                else:
                    converged = True
            # gpy.write_mesh(f"pre_remesh.obj", V, F)
            if remeshing:
                if (not use_features):
                    feature_vertices = np.array([],dtype=np.int32)

                if (output_sensitive and F_invalid.shape[0] > 0):
                    # we find the invalid faces
                    F_invalid = np.unique(I[invalid_U])
                    # We compute the face adjacency matrix
                    TT = face_adjacency(F)
                    # We find the set of invalid faces and their neighbors
                    F_invalid_neighbors = np.unique(TT[F_invalid,:].ravel())
                    # also add the invalid faces
                    F_invalid_neighbors = np.unique(np.hstack((F_invalid_neighbors,F_invalid)))

                    F_invalid_neighbors = one_ring(F,F_invalid)
                    # do another round of one ring
                    F_invalid_neighbors = one_ring(F,F_invalid_neighbors)
                    # We find the set of invalid vertices
                    # F_active = F[F_invalid_neighbors,:]
                    V_active, F_active, _, _ = gpy.remove_unreferenced(V, F[F_invalid_neighbors,:],return_maps=True)

                    if (F_active.shape[0] < F.shape[0] and F_active.shape[0] > 0):
                        # set of inactive faces
                        F_inactive = np.setdiff1d(np.arange(F.shape[0]), F_invalid_neighbors)
                        # set of inactive vertices
                        V_inactive, F_inactive, _, _ = gpy.remove_unreferenced(V, F[F_inactive,:],return_maps=True)
                        # Remesh only the active part
                        
                        V_active, F_active = remesh(V_active, F_active, i=remesh_iterations, h=h, project = True)
                        # We merge the active and inactive parts
                        V = np.vstack((V_active, V_inactive))
                        F = np.vstack((F_active, F_inactive + V_active.shape[0]))
                        # We remove the duplicate vertices
                        V,_,_,F = gpy.remove_duplicate_vertices(V,faces=F,epsilon=np.sqrt(np.finfo(V.dtype).eps))
                    else:
                        V, F = remesh(V, F, i=remesh_iterations, h=h, project = True)
                        # V, F = remesh(V, F, i=remesh_iterations, h=h, project = True, feature=feature_vertices)
                else:
                    V, F = remesh(V, F, i=remesh_iterations, h=h, project = True, feature=feature_vertices)
                    V_active, F_active = None, None
                    V_inactive, F_inactive = None, None
                    # V, F = remesh(V, F, i=2, h=h, project = True)
            else:
                V_active, F_active = None, None
                V_inactive, F_inactive = None, None
            # gpy.write_mesh(f"after_remesh.obj", V, F)
        elif (not stopped):
            # print("Done!")
            stopped = True

        if callback is not None:
            callback({'V':V, 'F':F,
                'V_active':V_active,
                'F_active':F_active,
                'V_inactive':V_inactive,
                'F_inactive':F_inactive,
                'its':its,
                'max_iter':max_iter,
                'U':U, 'S':S,
                'converged':converged,
                'resample_counter':resample_counter})

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
                if V_active is not None and F_active is not None:
                    visualize_full = False
                    plot_edges(V_active,F_active,'b-')
                    plt.plot(V_active[:,0],V_active[:,1],'b.')
                if V_inactive is not None and F_inactive is not None:
                    visualize_full = False
                    plot_edges(V_inactive,F_inactive,'y-')
                    plt.plot(V_inactive[:,0],V_inactive[:,1],'y.')
                if visualize_full and V is not None and F is not None:
                    plot_edges(V,F,'b-')
                    plt.plot(V[:,0],V[:,1],'b.')
                plt.draw()
                plt.pause(0.01)
            elif dim==3:
                if stopped:
                    # This mess is so that we can render something from polyscope in the same script, otherwise this callback will keep executing and deleting everything you plot.
                    if active_ps is not None:
                        active_ps.remove()
                        active_ps = None
                    if inactive_ps is not None:
                        inactive_ps.remove()
                        inactive_ps = None
                    if full_ps is not None:
                        full_ps.remove()
                        full_ps = None
                    # polyscope.remove_all_structures()
                else:
                    # cloud_U = polyscope.register_point_cloud("SDF evaluation points", U)
                    # cloud_U.add_scalar_quantity("How unhappy?", np.abs(g), enabled=True)
                    visualize_full = True
                    if V_active is not None and F_active is not None:
                        visualize_full = False
                        active_ps = polyscope.register_surface_mesh("active", V_active, F_active)
                    if V_inactive is not None and F_inactive is not None:
                        visualize_full = False
                        inactive_ps = polyscope.register_surface_mesh("inactive", V_inactive, F_inactive)
                    if visualize_full and V is not None and F is not None:
                        full_ps = polyscope.register_surface_mesh("full", V, F)

    if visualize and dim==3:
        polyscope.init()
        # polyscope.register_surface_mesh("gt",V2,F2)
        def polyscope_callback():
            run_flow_iteration()
        polyscope.set_user_callback(polyscope_callback)
        polyscope.show()
    else:
        while (its<max_iter and (not converged)):
            run_flow_iteration()

    if return_U:
        return V, F, U
    else:
        return V, F






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
        TT,_ = gpy.triangle_triangle_adjacency(F)
        return TT


def laplacian(v,f):
    dim = v.shape[1]
    if dim==3:
        L = gpy.cotangent_laplacian(v,f)
    elif dim==2:
        G = gpy.grad(v,f)
        A = 0.5*gpy.doublearea(v,f)
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
        return gpy.per_face_normals(V,F,unit_norm=unit_norm)
    
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
        α = gpy.tip_angles(V,F)
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
        l_sq = gpy.halfedge_lengths_squared(V,F)
        l_sq = np.maximum(l_sq, 100.*np.finfo(V.dtype).eps)
        M = gpy.massmatrix_intrinsic(l_sq,F,n=V.shape[0]) 
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
        V,F = gpy.remesh_botsch(V, F,
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
        V,F,I,J = gpy.remove_unreferenced(V, F, return_maps=True)
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

    _,I,b = gpy.squared_distance(V,V0,F0,use_cpp=True,use_aabb=True)
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
    P,I,_ = gpy.random_points_on_mesh(V, F, trial_n, rng=rng, return_indices=True)
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





