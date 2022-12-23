import numpy as np
from scipy.sparse import diags, eye, block_diag, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, bicg, splu
from scipy.ndimage.filters import gaussian_filter
from .fd_grad import fd_grad
from .fd_interpolate import fd_interpolate
from .matrix_from_function import matrix_from_function
from .compactly_supported_normal import compactly_supported_normal
from .grid_neighbors import grid_neighbors
from .grid_laplacian_eigenfunctions import grid_laplacian_eigenfunctions

def poisson_surface_reconstruction(P,N,gs=None,h=None,corner=None,stochastic=False,sigma_n=0.0,sigma=0.05,solve_subspace_dim=0,verbose=False,prior_fun=None):
    """
    Runs Poisson Surface Reconstruction from a set of points and normals to output a scalar field that takes negative values inside the surface and positive values outside the surface.
    
    Parameters
    ----------
    P : (n,dim) numpy array
        Coordinates of points in R^dim
    N : (n,dim) numpy array
        (Unit) normals at each point
    gs : (dim,) numpy array
        Number of grid points in each dimension
    h : (dim,) numpy array
        Grid spacing in each dimension
    corner : (dim,) numpy array
        Coordinates of the lower left corner of the grid
    stochastic : bool, optional (default False)
        Whether to use "Stochastic Poisson Surface Reconstruction" to output a mean and variance scalar field instead of just one scalar field
    sigma_n : float, optional (default 0.0)
        Noise level in the normals
    sigma : float, optional (default 0.05)
        Scalar global variance parameter
    solve_subspace_dim : int, optional (default 0)
        If > 0, use a subspace solver to solve the linear system. This is useful for large problems and essential in 3D.
    verbose : bool, optional (default True)
        Whether to print progress
    prior_fun : function, optional (default None)
        Function that takes a (m,dim) numpy array and returns a (m,dim) numpy array. This is used to specify a prior on the gradient of the scalar field (see Sec. 5 in "Stochastic Poisson Surface Reconstruction").
    
    Returns
    -------
    scalar_mean : (gs[0],gs[1],...,gs[dim-1]) numpy array
        Mean of the reconstructed scalar field
    scalar_variance : (gs[0],gs[1],...,gs[dim-1]) numpy array
        Variance of the reconstructed scalar field. This will only be part of the output if stochastic=True.
    grid_vertices : list of (gs[0],gs[1],...,gs[dim-1],dim) numpy arrays
        Grid vertices (each element in the list is one dimension), as in the output of np.meshgrid

    
    Notes
    -----
    See [this jupyter notebook](https://colab.research.google.com/drive/1DOXbDmqzIygxoQ6LeX0Ewq7HP4201mtr?usp=sharing) for a full tutorial on how to use this function.

    See also
    --------
    fd_interpolate, fd_grad, matrix_from_function, compactly_supported_normal, grid_neighbors

    Examples
    --------
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from gpytoolbox.poisson_surface_reconstruction import poisson_surface_reconstruction, random_points_on_polyline, png2poly
    # Generate random points on a polyline
    poly = gpytoolbox.png2poly("test/unit_tests_data/illustrator.png")[0]
    poly = poly - np.min(poly)
    poly = poly/np.max(poly)
    poly = 0.5*poly + 0.25
    poly = 3*poly - 1.5
    num_samples = 40
    np.random.seed(2)
    EC = edge_indices(poly.shape[0],closed=False)
    P,I,_ = random_points_on_mesh(poly, EC, num_samples, return_indices=True)
    vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
    vecs /= np.linalg.norm(vecs, axis=1)[:,None]
    J = np.array([[0., -1.], [1., 0.]])
    N = vecs @ J.T
    N = N[I,:]


    # Problem parameters
    gs = np.array([50,50])
    # Call to PSR
    scalar_mean, scalar_var, grid_vertices = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=0,verbose=True)
    
    # The probability of each grid vertex being inside the shape 
    prob_out = 1 - norm.cdf(scalar_mean,0,np.sqrt(scalar_var))

    gx = grid_vertices[0]
    gy = grid_vertices[1]

    # Plot mean and variance side by side with colormap
    fig, ax = plt.subplots(1,3)
    m0 = ax[0].pcolormesh(gx,gy,np.reshape(scalar_mean,gx.shape), cmap='RdBu',shading='gouraud', vmin=-np.max(np.abs(scalar_mean)), vmax=np.max(np.abs(scalar_mean)))
    ax[0].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
    q0 = ax[0].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
    ax[0].set_title('Mean')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(m0, cax=cax, orientation='vertical')

    m1 = ax[1].pcolormesh(gx,gy,np.reshape(np.sqrt(scalar_var),gx.shape), cmap='plasma',shading='gouraud')
    ax[1].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
    q1 = ax[1].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
    ax[1].set_title('Variance')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(m1, cax=cax, orientation='vertical')

    m2 = ax[2].pcolormesh(gx,gy,np.reshape(prob_out,gx.shape), cmap='viridis',shading='gouraud')
    ax[2].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
    q2 = ax[2].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
    ax[2].set_title('Probability of being inside')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(m2, cax=cax, orientation='vertical')
    plt.show()
    """

    

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
        # print(gs)
        # print(gs)
    elif ((h is None) or (corner is None)):
        h = (np.max(envelope_mult*P,axis=0) - np.min(envelope_mult*P,axis=0))/gs
        corner = np.min(envelope_mult*P,axis=0)
    assert(gs.shape[0] == dim)

    grid_length = gs*h
    # This is grid we will obtain the final values on
    grid_vertices = np.meshgrid(*[np.linspace(corner[dd], corner[dd] + (gs[dd]-1)*h[dd], gs[dd]) for dd in range(dim)])

    # Kernel function for the Gaussian process
    def kernel_fun(X,Y):
        return compactly_supported_normal(X-Y,n=3,sigma=1.5*np.min(h))
    # np.meshgrid(*[np.linspace(corner_dd[dd], corner_dd[dd] + (gs_dd[dd]-1)*h[dd], gs_dd[dd]) for dd in range(dim)])

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
            staggered_grid_vertices = np.meshgrid(*[np.linspace(corner_dd[dd], corner_dd[dd] + (gs_dd[dd]-1)*h[dd], gs_dd[dd]) for dd in range(dim)])
            staggered_grid_vertices = np.array(staggered_grid_vertices).reshape(dim, -1).T
        elif dim==3:
            staggered_grid_vertices = np.meshgrid(*[np.linspace(corner_dd[dd], corner_dd[dd] + (gs_dd[dd]-1)*h[dd], gs_dd[dd]) for dd in range(dim)],indexing='ij')
            staggered_grid_vertices = np.array(staggered_grid_vertices).reshape(dim, -1,order='F').T
        
        if verbose:
            t00 = time.time()

        ### Step 1.1.: Compute the matrix k1, which has size prod(gs_dd) x prod(gs_dd) and contains the kernel evaluations between all pairs of points in the staggered grid.
        # We could compute this matrix easily, by running
        # k1_slow = matrix_from_function(kernel_fun, staggered_grid_vertices, staggered_grid_vertices)
        # However, this would be extremely slow, O(prod(gs_dd)^2). Instead, we use the fact that the kernel is compactly supported, and only compute the kernel evaluations between pairs of points that are within a second-order neighborhood of each other.

        # Find the neighbors of each point in the staggered grid
        neighbor_rows = grid_neighbors(gs_dd,include_diagonals=True,include_self=True, order=2)
        # Find one cell with no out of bounds neighbors
        min_ind = np.min(neighbor_rows,axis=0)
        valid_ind = np.argwhere(min_ind>=0)[0]
        # What is the center of said cell
        center_sample_cell = staggered_grid_vertices[valid_ind,:]
        # And the coordinates of its second-order neighbors
        neighbors_sample_cell = staggered_grid_vertices[np.squeeze(neighbor_rows[:,valid_ind]),:]
        # Evaluate the kernel function just for this cell
        center_sample_cell_tiled = np.tile(center_sample_cell,(neighbors_sample_cell.shape[0],1))
        values_sample_cell = kernel_fun(center_sample_cell_tiled,neighbors_sample_cell)
        # Thanks to the grid structure, the values for every cell will be the same, so we can just tile the values_sample_cell vector
        V = np.tile(values_sample_cell,(np.prod(gs_dd),1)).T
        I = np.tile(np.arange(np.prod(gs_dd)), (neighbor_rows.shape[0],1))
        J = neighbor_rows
        # Remove out-of-bounds indices
        V[J<0] = 0
        J[J<0] = 0
        # And build k1:
        k1_fast = csr_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(np.prod(gs_dd),np.prod(gs_dd)))
        k1 = k1_fast
        if verbose:
            print("Time to compute k1: ", time.time() - t00)
            t00 = time.time()

        ### Step 1.2: Building k2, the matrix of kernel evaluations between the points in the staggered grid and the points in P. Once again, we could compute this easily with
        # k2_slow = matrix_from_function(kernel_fun, P, staggered_grid_vertices)
        # But this would be very slow, of order O(N*prod(gs_dd)). Instead, we use the grid structure to compute the cell that each point in P falls into and then only evaluate the kernel on said cell and its third-order neighborhood.


        # Find the cell that P falls into
        P_cells = np.floor((P - np.tile(corner_dd,(P.shape[0],1))) / np.tile(h,(P.shape[0],1))).astype(int)
        if dim==2:
            P_cells = np.ravel_multi_index((P_cells[:,0], P_cells[:,1]), dims=gs_dd,order='F')
        else:
            P_cells = np.ravel_multi_index((P_cells[:,0], P_cells[:,1], P_cells[:,2]), dims=gs_dd,order='F')

        # Find the neighbors of each P-associated cell
        order_neighbors = 3
        neighbors = np.arange(-order_neighbors,order_neighbors+1,dtype=int)
        for dd2 in range(dim-1):
            neighbors_along_dim =np.arange(-order_neighbors,order_neighbors+1,dtype=int)*np.prod(gs_dd[:dd2+1])
            previous_neighbors = np.kron(neighbors,np.ones(neighbors_along_dim.shape[0],dtype=int))
            neighbors_along_dim_repeated = np.kron(np.ones(neighbors.shape[0],dtype=int),neighbors_along_dim)
            neighbors = previous_neighbors + neighbors_along_dim_repeated
        # The neighbors give us the sparsity pattern of k2
        I = np.tile(np.arange(P.shape[0]),(neighbors.shape[0],1)).T
        J = np.tile(P_cells,(neighbors.shape[0],1)).T + np.tile(neighbors,(P_cells.shape[0],1))
        valid_indices = (J>=0)*(J<np.prod(gs_dd))
        J = J[valid_indices]
        I = I[valid_indices]

        # We evaluate the kernel function to get k2, but we do it with a known sparsity pattern, so this is O(P.shape[0]) instead of O(P.shape[0]*prod(gs_dd)).
        k2_fast = matrix_from_function(kernel_fun, P, staggered_grid_vertices,sparsity_pattern=[I,J])
        k2 = k2_fast
        if verbose:
            print("Time to compute k2: ", time.time() - t00)

        ### Step 1.3: Build K3, the matrix of kernel evaluations between sample points. If we did a true GP, we would compute this with
        # K3 = matrix_from_function(kernel_fun, P, P)
        # In reality, we approximate K3 with a lumped covariance matrix:
        # K3 = diags(np.squeeze(sampling_density)) + sigma_n*sigma_n*eye(P.shape[0])
        # But we really only need its inverse:
        K3_inv = diags(1/(np.squeeze(sampling_density) + sigma_n*sigma_n))
        

        ### Step 1.4: Solve for the mean and covariance of the vector field:
        if (prior_fun is None):
            means.append(k2.T @ K3_inv @ N[:,dd][:,None])
        else:
            # print(prior_fun(staggered_grid_vertices)[:,dd][:,None])
            # print(prior_fun(staggered_grid_vertices)[:,dd][:,None].shape)
            # print(N[:,dd][:,None].shape)
            means.append(prior_fun(staggered_grid_vertices)[:,dd][:,None] + k2.T @ K3_inv @ (N[:,dd][:,None] - prior_fun(P)[:,dd][:,None]))
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
    # Get the gradient so that its transpose is the divergence
    G = fd_grad(gs=gs,h=h)
    # Compute the divergence
    mean_divergence = G.T @ vector_mean
    # Build Laplacian
    L = G.T @ G
    # Solve for the mean scalar field
    mean_scalar, info = bicg(L + eps*eye(L.shape[0]),mean_divergence,atol=1e-10)
    assert(info==0)
    # Shift values of mean
    W = fd_interpolate(P,gs=gs,h=h,corner=corner)
    shift = np.sum((W @ mean_scalar) ) / P.shape[0]
    mean_scalar = mean_scalar - shift
    # ...and we're done, mean_scalar is computed
    if verbose:
        print("Time to compute mean PDE solution: ", time.time() - t10)
        t10 = time.time()


    if stochastic:
        # In this case, we want to compute the variances of the scalar field        
        cov_divergence = G.T @ vector_cov @ G


        
        if (solve_subspace_dim>0):
            # Project the covariance matrix onto a subspace (fast)
            if verbose:
                print("Solving for covariance on grid using subspace method")
                t20 = time.time()
            # _, vecs = eigenfunctions_laplacian(solve_subspace_dim,gs,grid_length)
            vecs = grid_laplacian_eigenfunctions(solve_subspace_dim,gs,grid_length)
            if verbose:
                print("Time to compute eigenfunctions: ", time.time() - t20)
                t20 = time.time()
            vecs = np.real(vecs)
            #vals = vecs.transpose() @ (L+0.0001*eye(L.shape[0])) @ vecs
            vals = np.sum(np.multiply(vecs.transpose()@(L+eps*eye(L.shape[0])),vecs.transpose()),axis=1)
            vals = csr_matrix(diags(vals))
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
            # Solve directly for the covariance on the grid (slow)
            if verbose:
                print("Solving for covariance directly")

            lu = splu(L+eps*eye(L.shape[0]))
            D = lu.solve(cov_divergence.toarray())
            cov_scalar = lu.solve(D.transpose())
            var_scalar = np.diag(cov_scalar)

        # Shift values of covariance (Q: is a constant shift enough?)
        var_scalar = var_scalar - np.min(var_scalar) + sigma_n*sigma_n + eps
        if verbose:
            print("Time to compute covariance PDE solution: ", time.time() - t10)       
        if verbose:
            print("Total step 2 time: ", time.time() - t1)
            print("Total time: ", time.time() - t0)
        return mean_scalar, var_scalar, grid_vertices
    else:
        if verbose:
            print("Total step 2 time: ", time.time() - t1)
            print("Total time: ", time.time() - t0)
        return mean_scalar, grid_vertices
