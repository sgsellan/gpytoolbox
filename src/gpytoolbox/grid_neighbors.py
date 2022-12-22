import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def grid_neighbors(gs,order=1,include_diagonals=False,include_self=False,output_unique=True):
    """
    Computes the n-th order neighbors of each cell in a grid.
    
    Parameters
    ----------
    gs : (dim,) numpy int array
        grid size in each dimension
    order: int, optional (default 1)
        neighborhood order (e.g., 1 means first-order neighbors)
    include_diagonals: bool, optional (default False).
        whether diagonal cells are considered to be neighbors
    include_self: bool, optional (default False)
        whether a cell is considered to be its own neighbor
    output_unique: bool, optional (default False)
        whether to output only unique neighbors (i.e., remove duplicates). This should only matter for order>=2 (for order=1, the output is always unique but this flag will change the ordering)
    
    Returns
    -------
    N : (num_neighbors, n) numpy int array
        The i-th column contains the list of neighbors of the i-th cell. Negative entries denote out-of-bounds cells.

    
    Examples
    --------
    gs = np.array([90,85])
    # This should be *only* the 8 neighbors at a distance of h
    N = gpytoolbox.grid_neighbors(gs, include_diagonals=False, include_self=False,order=1)
    # Now in each column of N, we have the indices of the four non-diagonal neighbors of the corresponding cell. For example, for the first (bottom left) cell, the neighbors are:
    N[:,0]
    # which should be two values of -1 (out of bounds), one value of 1 (the cell to the right), and one value of 85 (the cell above).
    """
    dim = gs.shape[0]
    cells = np.arange(np.prod(gs)) # all cell indices

    if dim==2:
        idy, idx = np.unravel_index(cells, gs,order='F')
        neigh_idx = np.vstack((idx-1, idx+1, idx, idx))
        neigh_idy = np.vstack((idy, idy, idy-1, idy+1))
        if include_diagonals:
            neigh_idx = np.vstack((neigh_idx,idx-1,idx-1,idx+1,idx+1))
            neigh_idy = np.vstack((neigh_idy,idy-1,idy+1,idy-1,idy+1))
        if include_self:
            neigh_idx = np.vstack((idx,neigh_idx))
            neigh_idy = np.vstack((idy,neigh_idy))

        out_of_bounds = np.logical_or(neigh_idx<0, neigh_idx>=gs[1]) | np.logical_or(neigh_idy<0, neigh_idy>=gs[0])
        
        # print(out_of_bounds[:,0])
        neighbor_rows = np.ravel_multi_index((neigh_idy, neigh_idx), dims=gs,mode='wrap',order='F')
    elif dim==3:
        idz, idy, idx = np.unravel_index(cells, gs, order='F')
        neigh_idx = np.vstack((idx-1, idx+1, idx, idx, idx, idx))
        neigh_idy = np.vstack((idy, idy, idy-1, idy+1, idy, idy))
        neigh_idz = np.vstack((idz, idz, idz, idz, idz-1, idz+1))
        if include_diagonals:
            # Indices of all 26 neighbors
            neigh_idx = np.vstack((neigh_idx,
                idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,
                idx  ,idx  ,       idx,idx,
                idx+1,idx+1,idx+1,idx+1,idx+1,idx+1,idx+1,idx+1))
            neigh_idy = np.vstack((neigh_idy,
                idy-1,idy-1,idy-1,idy,idy,idy+1,idy+1,idy+1,
                idy-1,idy-1,    idy+1,idy+1,
                idy-1,idy-1,idy-1,idy,idy,idy+1,idy+1,idy+1))
            neigh_idz = np.vstack((neigh_idz,
                idz-1,idz,idz+1,idz-1,idz+1,idz-1,idz,idz+1,
                idz-1,idz+1,idz-1,idz+1,
                idz-1,idz,idz+1,idz-1,idz+1,idz-1,idz,idz+1))
        if include_self:
            neigh_idx = np.vstack((idx,neigh_idx))
            neigh_idy = np.vstack((idy,neigh_idy))
            neigh_idz = np.vstack((idz,neigh_idz))
        
        out_of_bounds = np.logical_or(neigh_idx<0, neigh_idx>=gs[2]) | np.logical_or(neigh_idy<0, neigh_idy>=gs[1]) | np.logical_or(neigh_idz<0, neigh_idz>=gs[0])
        neighbor_rows = np.ravel_multi_index((neigh_idz, neigh_idy, neigh_idx), dims=gs,mode='wrap',order='F')


    current_order = 1
    while current_order < order:
        # Switching to a list made this 10x faster in 3D because of the memory allocation magic!
        neighbor_rows_list = [neighbor_rows]
        out_of_bounds_list = [out_of_bounds]
        # neighbor_rows_bigger = neighbor_rows
        # out_of_bounds_bigger = out_of_bounds
        for i in range(neighbor_rows.shape[0]):
            new_out_of_bounds = np.zeros(neighbor_rows.shape,dtype=bool)
            new_out_of_bounds[:,out_of_bounds[i,:]] = True
            new_out_of_bounds = new_out_of_bounds | out_of_bounds[:,neighbor_rows[i,:]]
            new_neighbor_rows = neighbor_rows[:,neighbor_rows[i,:]]
            # neighbor_rows_bigger = np.vstack((neighbor_rows_bigger,new_neighbor_rows))
            neighbor_rows_list.append(new_neighbor_rows)
            # out_of_bounds_bigger = np.vstack((out_of_bounds_bigger,new_out_of_bounds))
            out_of_bounds_list.append(new_out_of_bounds)
        neighbor_rows = np.vstack(neighbor_rows_list)
        out_of_bounds = np.vstack(out_of_bounds_list)
        # neighbor_rows = neighbor_rows_bigger
        # out_of_bounds = out_of_bounds_bigger
        current_order += 1

    if output_unique:
        unique_ind = np.unique(neighbor_rows[:,0],return_index=True)[1]
        neighbor_rows = neighbor_rows[unique_ind,:]
        out_of_bounds = out_of_bounds[unique_ind,:]


    neighbor_rows[out_of_bounds] = -1
    # neighbor_rows = neighbor_rows[unique_ind,:]
    return neighbor_rows
