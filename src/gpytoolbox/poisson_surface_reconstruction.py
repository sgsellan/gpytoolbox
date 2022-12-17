import numpy as np
from scipy.sparse import diags, eye, block_diag
from scipy.sparse.linalg import spsolve, bicg, splu
from scipy.ndimage.filters import gaussian_filter
from .fd_grad import fd_grad
from .fd_interpolate import fd_interpolate
from .matrix_from_function import matrix_from_function
from .compactly_supported_normal import compactly_supported_normal

def poisson_surface_reconstruction(P,N,gs=None,h=None,corner=None,screened=False,sigma_n=0.0,sigma=0.1):

    # To do: set gs, h, corner to something that makes sense given P if all/some are None
    def kernel_fun(X,Y):
        return compactly_supported_normal(X-Y,n=3,sigma=0.1)

    dim = P.shape[1]
    assert(dim == N.shape[1])

    # Estimate sampling density at each point in P
    W_weights = fd_interpolate(P,gs=(gs+1),h=h,corner=(corner-0.5*h))
    image = (W_weights.T @ np.ones((N.shape[0],1)))/(np.prod(h))
    image_blurred = gaussian_filter(np.reshape(image,gs+1,order='F'),sigma=3)
    image_blurred_vectorized = np.reshape(image_blurred,(np.prod(gs+1),1),order='F')
    sampling_density = W_weights @ image_blurred_vectorized
        
        

    # Step 1: Gaussian process interpolation from N to a regular grid
    means = []
    covs = []
    for dd in range(dim):
        # Build a staggered grid in the dd-th dimension
        corner_dd = corner.copy()
        corner_dd[dd] += 0.5*h[dd]
        gs_dd = gs.copy()
        gs_dd[dd] -= 1
        # generate grid vertices of dimension dim
        grid_vertices = np.meshgrid(*[np.linspace(corner_dd[dd], corner_dd[dd] + (gs_dd[dd]-1)*h[dd], gs_dd[dd]) for dd in range(dim)])
        grid_vertices = np.array(grid_vertices).reshape(dim, -1).T
        # Compute k2, the kernel matrix between the points in P and the points in the regular grid
        k1 = matrix_from_function(kernel_fun, grid_vertices, grid_vertices)
        k2 = matrix_from_function(kernel_fun, P, grid_vertices)
        # This is what we'd get if we did a true GP:
        # K3 = matrix_from_function(kernel_fun, P, P)
        # In reality, we approximate K3 with a lumped covariance matrix:
        
        K3 = diags(np.squeeze(sampling_density)) + sigma_n*sigma_n*eye(P.shape[0])
        K3_inv = diags(1/(np.squeeze(sampling_density) + sigma_n*sigma_n))
        

        # Solve for the mean and covariance:
        # Debugging matrix sizes:
        # print(k1.shape)
        # print(k2.T.shape)
        # print(K3_inv.shape)
        # print(N[:,dd][:,None].shape)
        means.append(k2.T @ K3_inv @ N[:,dd][:,None])
        covs.append(k1 - k2.T @ K3_inv @ k2)

    # Concatenate the mean and covariance matrices
    vector_mean = np.concatenate(means)
    vector_cov = sigma*sigma*block_diag(covs)

    # Step 2: Solve Poisson equation on the regular grid
    
    G = fd_grad(gs=gs,h=h)
    mean_divergence = G.T @ vector_mean
    L = G.T @ G
    mean_scalar, info = bicg(L + 0.0001*eye(L.shape[0]),mean_divergence,atol=1e-10)
    assert(info==0)
    # Shift values of mean
    W = fd_interpolate(P,gs=gs,h=h,corner=corner)
    shift = np.sum((W @ mean_scalar) ) / P.shape[0]
    mean_scalar = mean_scalar - shift


    cov_divergence = G.T @ vector_cov @ G
    lu = splu(L+0.0001*eye(L.shape[0]))
    D = lu.solve(cov_divergence.toarray())
    cov_scalar = lu.solve(D.transpose())
    #cov = np.diag(spsolve(L+0.0001*eye(L.shape[0]),csc_matrix(D.transpose())).toarray())
    var_scalar = np.diag(cov_scalar)
    # Shift values of covariance
    var_scalar = var_scalar - np.min(var_scalar) + sigma_n*sigma_n
    
    return mean_scalar, var_scalar