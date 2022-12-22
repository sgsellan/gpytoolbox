import numpy as np

# To-do: this should output values too.

def grid_laplacian_eigenfunctions(num_modes,gs,l):
    """
    Returns the eigenfunctions and eigenvalues of the Laplacian on a regular grid with Neumann boundary conditions.
    
    Parameters
    ----------
    num_modes : int
        Number of eigenfunctions and eigenvalues to return
    gs : (dim,) int numpy array
        Grid size in each dimension
    l : (dim,) float numpy array
        Length of the grid in each dimension
    
    Returns
    -------
    vecs : (num_grid_points,num_modes) numpy array
        Eigenfunctions

    See also
    --------
    poisson_surface_reconstruction

    Notes
    -----
    This function uses the formula in "Eigenvalues of the Laplacian with Neumann Boundary Conditions" by H. P. W. Gottlieb, 1985. Right now, this function only works for 3D and 2D grids. It should be easy/trivial to extend to higher dimensions.

    Examples
    --------
    import numpy as np
    import gpytoolbox
    gs = np.array([10,10])
    l = np.array([1.0,1.0])
    num_modes = 10
    vecs, vals = gpytoolbox.grid_laplacian_eigenfunctions(num_modes,gs,l)
    """

    # There's probably some refactoring that could be done on this code so that the dimensionality is not hard-coded.

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
        # This is ad-hoc so it's faster... We should come up with a smarter way of doing this.
        num_in_each_dim = round(np.sqrt(num_modes//8))
        # num_in_each_dim = num_modes
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
        vals = vals[relevant_indices]
    else:
        gx, gy = np.meshgrid(np.linspace(0,l[0],gs[0]),np.linspace(0,l[1],gs[1]))
        # h = np.array([gx[1,1]-gx[0,0],gy[1,1]-gy[0,0]])
        v = np.concatenate((np.reshape(gx,(-1, 1)),np.reshape(gy,(-1, 1))),axis=1)
        num_in_each_dim = num_modes // 10 
        num_in_each_dim = num_modes
        vals = np.zeros(num_in_each_dim*num_in_each_dim)

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
        # print(I_vector[relevant_indices])
        # print(J_vector[relevant_indices])
        vecs = vecs_debug
        vals = vals[relevant_indices]
    # This is not necessary
    # vecs = vecs/np.tile(np.linalg.norm(vecs,axis=0),(vecs.shape[0],1))
    # print(np.linalg.norm(vecs,axis=0),(vecs.shape[0],1))

    return vecs
