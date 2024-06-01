import numpy as np

def particle_swarm(fun,lb,ub,n_particles=100,max_iter=100,momentum=0.9,phi=0.1,verbose=False, topology='full'):
    """Particle swarm optimization.
    
    Parameters
    ----------
    fun : callable
        Function to minimize
    lb : (n,) numpy double array
        Vector of lower bounds
    ub : (n,) numpy double array
        Vector of upper bounds
    n_particles : int, optional (default: 100)
        Number of particles
    max_iter : int, optional (default: 1000)
        Maximum number of iterations
    topology : str, optional (default: 'full')
        Connectivity of the swarm. Either 'full' or 'ring'.
    verbose : bool, optional (default: False)
        Print progress to stdout
    
    Returns
    -------
    x : (n,) numpy double array
        Best solution
    f : double
        Best objective value
    """
    current_best_f = np.inf
    n = len(lb)
    x = np.random.uniform(lb,ub,(n_particles,n))
    best_xi = x.copy()
    best_fi = np.inf*np.ones(n_particles)
    for i in range(n_particles):
        # This x
        xi = x[i,:]
        # print(xi)
        f = fun(xi)
        # print(xi)
        best_xi[i,:] = xi.copy()
        best_fi[i] = np.squeeze(f.copy())
        # if verbose:
            # print("Particle %d: f = %f" % (i,f))
        if f < current_best_f:
            current_best_x = xi.copy()
            current_best_f = f.copy()
    # print(current_best_x.reshape((-1, 3)))
    # initialize particle velocities
    velocity_lb = lb - ub
    velocity_ub = ub - lb
    v = np.random.uniform(velocity_lb,velocity_ub,(n_particles,n))

    # Repeat until convergence
    for iter in range(max_iter):
        for i in range(n_particles):
            # Pick random numbers rp and rg between 0 and 1
            rp = np.random.uniform()
            rg = np.random.uniform()
            # Update velocity
            if topology == 'full':
                best_neighbor = current_best_x.copy()
            elif topology == 'ring':
                # Pick the best among the current particle and its two neighbors
                neighbors = np.array([i-1,i,i+1])%n_particles
                xi_neighbors = np.array([best_xi[j,:] for j in neighbors])
                fi_neighbors = np.array([best_fi[j] for j in neighbors])
                best_neighbor = xi_neighbors[np.argmin(fi_neighbors)]
            v[i,:] = momentum*v[i,:] + phi*rp*(best_xi[i,:] - x[i,:]) + phi*rg*(best_neighbor - x[i,:])
            # Update position
            x[i,:] = x[i,:] + v[i,:]
            # Check bounds
            x[i,:] = np.maximum(x[i,:],lb)
            x[i,:] = np.minimum(x[i,:],ub)
            # Evaluate objective
            f = fun(x[i,:])
            # Update best position
            if f < best_fi[i]:
                best_xi[i,:] = x[i,:].copy()
                best_fi[i] = np.squeeze(f.copy())
                if f < current_best_f:
                    current_best_x = x[i,:].copy()
                    current_best_f = f.copy()
        if verbose:
            print("Iteration %d: f = %f" % (iter,current_best_f))
    # print(current_best_x)
    return current_best_x, current_best_f
    
