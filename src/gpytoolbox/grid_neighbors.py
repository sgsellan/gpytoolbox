import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def grid_neighbors(gs,order=1,include_diagonals=False,include_self=False,output_unique=False):
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
        whether to output only unique neighbors (i.e., remove duplicates). This should only matter for order>=2
    
    Returns
    -------
    N : (num_neighbors, n) numpy int array
        The i-th column contains the list of neighbors of the i-th cell. Negative entries denote out-of-bounds cells.

    
    Examples
    --------
    TODO
    """
    dim = gs.shape[0]
    cells = np.arange(np.prod(gs)) # all cell indices

    if dim==2:
        idy, idx = np.unravel_index(cells, gs,order='F')
        if include_diagonals:
            neigh_idx = np.vstack((idx-1,idx-1,idx-1,idx ,idx   ,idx+1,idx+1,idx+1))
            neigh_idy = np.vstack((idy-1,idy  ,idy+1,idy-1,idy+1,idy-1,idy  ,idy+1))
        else:
            neigh_idx = np.vstack((idx-1, idx+1, idx, idx))
            neigh_idy = np.vstack((idy, idy, idy-1, idy+1))
        if include_self:
            neigh_idx = np.vstack((neigh_idx,idx))
            neigh_idy = np.vstack((neigh_idy,idy))

        out_of_bounds = np.logical_or(neigh_idx<0, neigh_idx>=gs[1]) | np.logical_or(neigh_idy<0, neigh_idy>=gs[0])
        
        # print(out_of_bounds[:,0])
        neighbor_rows = np.ravel_multi_index((neigh_idy, neigh_idx), dims=gs,mode='wrap',order='F')
    elif dim==3:
        idz, idy, idx = np.unravel_index(cells, gs, order='F')
        if include_diagonals:
            # Indices of all 26 neighbors
            neigh_idx = np.vstack((
                idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,idx-1,
                idx,idx,idx,idx,idx,idx,idx,idx,
                idx+1,idx+1,idx+1,idx+1,idx+1,idx+1,idx+1,idx+1,idx+1))
            neigh_idy = np.vstack((
                idy-1,idy-1,idy-1,idy,idy,idy,idy+1,idy+1,idy+1,
                idy-1,idy-1,idy-1,idy,idy,    idy+1,idy+1,idy+1,
                idy-1,idy-1,idy-1,idy,idy,idy,idy+1,idy+1,idy+1))
            neigh_idz = np.vstack((
                idz-1,idz,idz+1,idz-1,idz,idz+1,idz-1,idz,idz+1,
                idz-1,idz,idz+1,idz-1,    idz+1,idz-1,idz,idz+1,
                idz-1,idz,idz+1,idz-1,idz,idz+1,idz-1,idz,idz+1))
        else:
            neigh_idx = np.vstack((idx-1, idx+1, idx, idx, idx, idx))
            neigh_idy = np.vstack((idy, idy, idy-1, idy+1, idy, idy))
            neigh_idz = np.vstack((idz, idz, idz, idz, idz-1, idz+1))
        if include_self:
            neigh_idx = np.vstack((neigh_idx,idx))
            neigh_idy = np.vstack((neigh_idy,idy))
            neigh_idz = np.vstack((neigh_idz,idz))
        
        out_of_bounds = np.logical_or(neigh_idx<0, neigh_idx>=gs[2]) | np.logical_or(neigh_idy<0, neigh_idy>=gs[1]) | np.logical_or(neigh_idz<0, neigh_idz>=gs[0])
        neighbor_rows = np.ravel_multi_index((neigh_idz, neigh_idy, neigh_idx), dims=gs,mode='wrap',order='F')

    # print(neighbor_rows)

    current_order = 1
    while current_order < order:
        # neighbor_rows = np.hstack((neighbor_rows,neighbor_rows[:,neighbor_rows>=0]))
        neighbor_rows_bigger = neighbor_rows
        out_of_bounds_bigger = out_of_bounds
        for i in range(neighbor_rows.shape[0]):
            new_out_of_bounds = np.zeros(neighbor_rows.shape,dtype=bool)
            new_out_of_bounds[:,out_of_bounds[i,:]] = True
            new_out_of_bounds = new_out_of_bounds | out_of_bounds
            neighbor_rows_bigger = np.vstack((neighbor_rows_bigger,neighbor_rows[:,neighbor_rows[i,:]]))
            out_of_bounds_bigger = np.vstack((out_of_bounds_bigger,new_out_of_bounds))
        neighbor_rows = neighbor_rows_bigger
        out_of_bounds = out_of_bounds_bigger
        current_order += 1

    if output_unique:
        unique_ind = np.unique(neighbor_rows[:,0],return_index=True)[1]
        neighbor_rows = neighbor_rows[unique_ind,:]
        out_of_bounds = out_of_bounds[unique_ind,:]

    neighbor_rows[out_of_bounds] = -1
    # neighbor_rows = neighbor_rows[unique_ind,:]
    return neighbor_rows
