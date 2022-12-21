import numpy as np
from scipy.sparse import diags, eye, block_diag, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, bicg, splu
from scipy.ndimage.filters import gaussian_filter
from .fd_grad import fd_grad
from .fd_interpolate import fd_interpolate
from .matrix_from_function import matrix_from_function
from .compactly_supported_normal import compactly_supported_normal
from .grid_neighbors import grid_neighbors


import matplotlib.pyplot as plt
# TO DO:
# - Test 3D functionality

def poisson_surface_reconstruction(P,N,gs=None,h=None,corner=None,stochastic=True,sigma_n=0.0,sigma=0.1,solve_subspace_dim=0,verbose=True):

    # Kernel function for the Gaussian process
    def kernel_fun(X,Y):
        return compactly_supported_normal(X-Y,n=3,sigma=np.mean(h))

    # Set problem parameters to values that make sense
    dim = P.shape[1]
    assert(dim == N.shape[1])

    
    if ((gs is None) and (h is None) and (corner is None)):
        # Default to a grid that is 100x100x...x100
        gs = 100*np.ones(dim,dtype=int)

    envelope_mult = 1.5 # how tightly we want to envelope the data
    if (gs is None):
        assert(h is not None)
        assert(corner is not None)
        gs = np.floor((np.max(envelope_mult*P,axis=0) - corner)/h).astype(int)
        print(gs)
        # print(gs)
    elif ((h is None) or (corner is None)):
        h = (np.max(envelope_mult*P,axis=0) - np.min(envelope_mult*P,axis=0))/gs
        corner = np.min(envelope_mult*P,axis=0)
    assert(gs.shape[0] == dim)

    grid_length = gs*h

    eps = 0.000001 # very tiny value to regularize matrix rank

    # Estimate sampling density at each point in P
    W_weights = fd_interpolate(P,gs=(gs+1),h=h,corner=(corner-0.5*h))
    image = (W_weights.T @ np.ones((N.shape[0],1)))/(np.prod(h))
    image_blurred = gaussian_filter(np.reshape(image,gs+1,order='F'),sigma=3)
    image_blurred_vectorized = np.reshape(image_blurred,(np.prod(gs+1),1),order='F')
    sampling_density = W_weights @ image_blurred_vectorized
        
    # Step 1: Gaussian process interpolation from N to a regular grid
    if verbose:
        print("Step 1: Gaussian process interpolation from N to a regular grid")
        # Log time to compute the kernel matrix
        import time
        t0 = time.time()

    means = []
    covs = []
    for dd in range(dim):
        # Build a staggered grid in the dd-th dimension
        corner_dd = corner.copy()
        corner_dd[dd] += 0.5*h[dd]
        gs_dd = gs.copy()
        gs_dd[dd] -= 1
        # generate grid vertices of dimension dim
        if dim==2:
            grid_vertices = np.meshgrid(*[np.linspace(corner_dd[dd], corner_dd[dd] + (gs_dd[dd]-1)*h[dd], gs_dd[dd]) for dd in range(dim)])
            grid_vertices = np.array(grid_vertices).reshape(dim, -1).T
        elif dim==3:
            grid_vertices = np.meshgrid(*[np.linspace(corner_dd[dd], corner_dd[dd] + (gs_dd[dd]-1)*h[dd], gs_dd[dd]) for dd in range(dim)],indexing='ij')
            grid_vertices = np.array(grid_vertices).reshape(dim, -1,order='F').T
        
        if verbose:
            t00 = time.time()
        # k1 is the kernel matrix between grid vertices, which we could compute with
        # k1 = matrix_from_function(kernel_fun, grid_vertices, grid_vertices)
        # but that takes a long to compute if we don't leverage that we know the sparsity pattern beforehand. In fact, we know that the only non-zero entries are of vertices and their second-level neighbors.
        # order_neighbors = 2
        # neighbors = np.arange(-order_neighbors,order_neighbors+1,dtype=int)
        # for dd2 in range(dim-1):
        #     neighbors_along_dim =np.arange(-order_neighbors,order_neighbors+1,dtype=int)*np.prod(gs_dd[:dd2+1])
        #     previous_neighbors = np.kron(neighbors,np.ones(neighbors_along_dim.shape[0],dtype=int))
        #     neighbors_along_dim_repeated = np.kron(np.ones(neighbors.shape[0],dtype=int),neighbors_along_dim)
        #     neighbors = previous_neighbors + neighbors_along_dim_repeated
        #     # neighbors = np.append(neighbors,neighbors*np.prod(gs[:dd2+1]))
        # all_grid_vertices = np.arange(np.prod(gs_dd))
        # # print(all_grid_vertices)
        # I = np.tile(all_grid_vertices,(neighbors.shape[0],1)).T
        # J = I + np.tile(neighbors,(all_grid_vertices.shape[0],1))
        # valid_indices = (J>=0)*(J<np.prod(gs_dd))
        # J = J[valid_indices]
        # I = I[valid_indices]

        # k1 = matrix_from_function(kernel_fun, grid_vertices, grid_vertices,sparsity_pattern=[I,J])

        # print("Slow:",time.time()-t00)
        # t00 = time.time()

        neighbor_rows = grid_neighbors(gs_dd,include_diagonals=True,include_self=True, order=1)
        # Find one cell with no out of bounds neighbors
        min_ind = np.min(neighbor_rows,axis=0)
        valid_ind = np.argwhere(min_ind>=0)[0]
        unique_ind = np.unique(neighbor_rows[:,valid_ind],return_index=True)[1]
        neighbor_rows = neighbor_rows[unique_ind,:]
        center_sample_cell = grid_vertices[valid_ind,:]
        neighbors_sample_cell = grid_vertices[np.squeeze(neighbor_rows[:,valid_ind]),:]
        center_sample_cell_tiled = np.tile(center_sample_cell,(neighbors_sample_cell.shape[0],1))
        values_sample_cell = kernel_fun(center_sample_cell_tiled,neighbors_sample_cell)
        V = np.tile(values_sample_cell,(np.prod(gs_dd),1)).T
        I = np.tile(np.arange(np.prod(gs_dd)), (neighbor_rows.shape[0],1))
        J = neighbor_rows
        V[J<0] = 0
        J[J<0] = 0
        k1_fast = csr_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(np.prod(gs_dd),np.prod(gs_dd)))
        # k1_fast = coo_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(np.prod(gs_dd),np.prod(gs_dd))).todia().tocsr()
        # print("Misc3:",time.time()-t00)
        # t00 = time.time()
        # k1 = k1_fast
        # print(neighbors_sample_cell.shape)
        # print(min_ind.shape)



 


        # I = np.tile(np.arange(np.prod(gs_dd)), (neighbor_rows.shape[0],1))
        # J = neighbor_rows
        # V = np.ones((neighbor_rows.shape[0],np.prod(gs_dd)))
        # V[J<0] = 0
        # J[J<0] = 0
        # A = coo_matrix(coo_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(np.prod(gs_dd),np.prod(gs_dd))))
        # print(A.max())
        # # print(A.diagonal())
        # # A = coo_matrix(A @ A)
        # I = A.row
        # J = A.col
        # k1_debug = matrix_from_function(kernel_fun, grid_vertices, grid_vertices,sparsity_pattern=[I,J])
        
        # assert(np.isclose(k1.toarray(),k1_debug.toarray(),atol=1e-5).all())
        # print(np.max(np.abs(k1-k1_debug)))
        # print(np.max(np.abs(k1-k1_fast)))
        # print("DISTANCE BETWEEN OLD MATRIX AND FAST MATRIX:", np.max(np.abs(k1-k1_fast)))
        k1 = k1_fast
        # k1 = k1_debug
        # assert((k1.toarray()==k1_debug.toarray()).all())

        # Could debug that we are building the proper matrix with this:
        # k1_debug = matrix_from_function(kernel_fun, grid_vertices, grid_vertices)
        # print(np.linalg.norm(k1.toarray()-k1_debug.toarray()))


        # Compute k2, the kernel matrix between the points in P and the points in the regular grid
        if verbose:
            print("Time to compute k1: ", time.time() - t00)
            t00 = time.time()
        # It's faster if we first find the cells that P falls into
        P_cells = np.floor((P - np.tile(corner,(P.shape[0],1))) / np.tile(h,(P.shape[0],1))).astype(int)
        if dim==2:
            P_cells = np.ravel_multi_index((P_cells[:,0], P_cells[:,1]), dims=gs_dd,order='F')
        else:
            P_cells = np.ravel_multi_index((P_cells[:,0], P_cells[:,1], P_cells[:,2]), dims=gs_dd,order='F')

        order_neighbors = 2
        neighbors = np.arange(-order_neighbors,order_neighbors+1,dtype=int)
        for dd2 in range(dim-1):
            neighbors_along_dim =np.arange(-order_neighbors,order_neighbors+1,dtype=int)*np.prod(gs_dd[:dd2+1])
            previous_neighbors = np.kron(neighbors,np.ones(neighbors_along_dim.shape[0],dtype=int))
            neighbors_along_dim_repeated = np.kron(np.ones(neighbors.shape[0],dtype=int),neighbors_along_dim)
            neighbors = previous_neighbors + neighbors_along_dim_repeated
        # # print(all_grid_vertices)
        I = np.tile(np.arange(P.shape[0]),(neighbors.shape[0],1)).T
        J = np.tile(P_cells,(neighbors.shape[0],1)).T + np.tile(neighbors,(P_cells.shape[0],1))
        valid_indices = (J>=0)*(J<np.prod(gs_dd))
        J = J[valid_indices]
        I = I[valid_indices]
        # Build I and J


        k2_debug = matrix_from_function(kernel_fun, P, grid_vertices,sparsity_pattern=[I,J])
        


        # k2 = matrix_from_function(kernel_fun, P, grid_vertices)
        # print("DISTANCE BETWEEN OLD MATRIX AND FAST MATRIX:", np.max(np.abs(k2-k2_debug)))
        k2 = k2_debug
        if verbose:
            print("Time to compute k2: ", time.time() - t00)
        # This is what we'd get if we did a true GP:
        # K3 = matrix_from_function(kernel_fun, P, P)
        # In reality, we approximate K3 with a lumped covariance matrix:
        # K3 = diags(np.squeeze(sampling_density)) + sigma_n*sigma_n*eye(P.shape[0])
        # But we really only need its inverse:
        K3_inv = diags(1/(np.squeeze(sampling_density) + sigma_n*sigma_n))
        

        # Solve for the mean and covariance of the vector field:
        means.append(k2.T @ K3_inv @ N[:,dd][:,None])
        if stochastic:
            covs.append(k1 - k2.T @ K3_inv @ k2)

    
    # Concatenate the mean and covariance matrices
    vector_mean = np.concatenate(means)
    if stochastic:
        vector_cov = sigma*sigma*block_diag(covs)

    if verbose:
        print("Total step 1 time: ", time.time() - t0)
    # Step 2: Solve Poisson equation on the regular grid
    

    if verbose:
        print("Step 2: Solve Poisson equation on the regular grid")
        t1 = time.time()
    if verbose:
        t10 = time.time()
    G = fd_grad(gs=gs,h=h)
    mean_divergence = G.T @ vector_mean
    L = G.T @ G
    mean_scalar, info = bicg(L + eps*eye(L.shape[0]),mean_divergence,atol=1e-10)
    assert(info==0)
    # Shift values of mean
    W = fd_interpolate(P,gs=gs,h=h,corner=corner)
    shift = np.sum((W @ mean_scalar) ) / P.shape[0]
    mean_scalar = mean_scalar - shift
    if verbose:
        print("Time to compute mean PDE solution: ", time.time() - t10)
        t10 = time.time()

    if stochastic:
        
        cov_divergence = G.T @ vector_cov @ G


        # Solve directly for the covariance on the grid (slow)
        if (solve_subspace_dim>0):
            if verbose:
                print("Solving for covariance on grid using subspace method")
                t20 = time.time()
            _, vecs = eigenfunctions_laplacian(solve_subspace_dim,gs,grid_length)
            if verbose:
                print("Time to compute eigenfunctions: ", time.time() - t20)
                t20 = time.time()
            vecs = np.real(vecs)
            #vals = vecs.transpose() @ (L+0.0001*eye(L.shape[0])) @ vecs
            vals = np.sum(np.multiply(vecs.transpose()@(L+eps*eye(L.shape[0])),vecs.transpose()),axis=1)
            vals = diags(vals)
            if verbose:
                print("Time to compute eigenvalues: ", time.time() - t20)
                t20 = time.time()
            B = (vecs.transpose()@cov_divergence.astype(np.float32))@vecs
            if verbose:
                print("Time to project problem onto subspace: ", time.time() - t20)
                t20 = time.time()
            
            D = spsolve(vals,B)
            #D = np.linalg.solve(vals,B)
            # cov_small = np.linalg.solve(vals,D.transpose())
            cov_small = spsolve(vals,D.transpose()).astype(np.float32)
            if verbose:
                print("Time to solve in subspace: ", time.time() - t20)
                t20 = time.time()
            var_scalar = np.sum(np.multiply(vecs@cov_small,vecs),axis=1)
            if verbose:
                print("Time to reproject to full space", time.time() - t20)
        else:
            if verbose:
                print("Solving for covariance directly")

            lu = splu(L+eps*eye(L.shape[0]))
            D = lu.solve(cov_divergence.toarray())
            cov_scalar = lu.solve(D.transpose())
            var_scalar = np.diag(cov_scalar)
        # 

        #cov = np.diag(spsolve(L+0.0001*eye(L.shape[0]),csc_matrix(D.transpose())).toarray())
        
        # Shift values of covariance
        var_scalar = var_scalar - np.min(var_scalar) + sigma_n*sigma_n
        if verbose:
            print("Time to compute covariance PDE solution: ", time.time() - t10)       
        if verbose:
            print("Total step 2 time: ", time.time() - t1)
            print("Total time: ", time.time() - t0)
        return mean_scalar, var_scalar
    else:
        if verbose:
            print("Total step 2 time: ", time.time() - t1)
            print("Total time: ", time.time() - t0)
        return mean_scalar


def eigenfunctions_laplacian(num_modes,gs,l):
    # Get grid
    dim = gs.shape[0]

    def psi(N,l,v):
        d = v.shape[1]
        out = np.ones(v.shape[0])
        val = 0
        for dd in range(d):
            out = out*np.cos(N[dd]*np.pi*v[:,dd]/l[dd])
            val = val + (np.pi**2.0)*((N[dd]/l[dd])**2.0)
        return out, val


    if dim==3:
        gx, gy, gz = np.meshgrid(np.linspace(0,l[0],gs[0]),np.linspace(0,l[1],gs[1]),np.linspace(0,l[2],gs[2]),indexing='ij')
        v = np.concatenate((np.reshape(gx,(-1, 1),order='F'),np.reshape(gy,(-1, 1),order='F'),np.reshape(gz,(-1, 1),order='F')),axis=1)
        #num_in_each_dim = num_modes // 10 # this should be enough
        num_in_each_dim = round(np.sqrt(num_modes//8))
        vals_debug = np.zeros(num_in_each_dim*num_in_each_dim*num_in_each_dim)
        K_vector = np.arange(num_in_each_dim*num_in_each_dim*num_in_each_dim) % num_in_each_dim
        J_vector = np.arange(num_in_each_dim*num_in_each_dim*num_in_each_dim) // num_in_each_dim % num_in_each_dim
        I_vector = np.arange(num_in_each_dim*num_in_each_dim*num_in_each_dim) // (num_in_each_dim*num_in_each_dim)

        ind_vectors = []
        ind_vectors.append(I_vector)
        ind_vectors.append(J_vector)
        ind_vectors.append(K_vector)
        for dd in range(dim):
            vals_debug = vals_debug + (np.pi**2.0)*((ind_vectors[dd]/l[dd])**2.0)
        # assert((vals_debug==vals).all())
        order = np.argsort(vals_debug)
        relevant_indices = order[0:num_modes]
        vecs_debug = np.ones((v.shape[0],num_modes))
        for s in range(len(relevant_indices)):
            vecs_debug[:,s], _ = psi([I_vector[relevant_indices[s]],J_vector[relevant_indices[s]],K_vector[relevant_indices[s]]],l,v)
        vecs = vecs_debug
        vals = vals_debug

    else:
        gx, gy = np.meshgrid(np.linspace(0,l[0],gs[0]),np.linspace(0,l[1],gs[1]))
        # h = np.array([gx[1,1]-gx[0,0],gy[1,1]-gy[0,0]])
        v = np.concatenate((np.reshape(gx,(-1, 1)),np.reshape(gy,(-1, 1))),axis=1)
        num_in_each_dim = num_modes // 10 # this should be enough
        # vecs = np.ones((v.shape[0],num_in_each_dim*num_in_each_dim))
        # vecs_debug = np.ones((v.shape[0],num_in_each_dim*num_in_each_dim))
        vals = np.zeros(num_in_each_dim*num_in_each_dim)
        # vals_debug = np.zeros(num_in_each_dim*num_in_each_dim)

        I_vector = np.arange(num_in_each_dim*num_in_each_dim)// num_in_each_dim
        
        J_vector = np.arange(num_in_each_dim*num_in_each_dim)% num_in_each_dim
        
        ind_vectors = []
        ind_vectors.append(I_vector)
        ind_vectors.append(J_vector)

        for dd in range(dim):
            vals = vals + (np.pi**2.0)*((ind_vectors[dd]/l[dd])**2.0)
        order = np.argsort(vals)
        relevant_indices = order[0:num_modes]
        vecs_debug = np.ones((v.shape[0],num_modes))
        for s in range(len(relevant_indices)):
            vecs_debug[:,s], _ = psi([I_vector[relevant_indices[s]],J_vector[relevant_indices[s]]],l,v)
        vecs = vecs_debug
    # This is not necessary
    # vecs = vecs/np.tile(np.linalg.norm(vecs,axis=0),(vecs.shape[0],1))
    # print(np.linalg.norm(vecs,axis=0),(vecs.shape[0],1))

    return vals, vecs
