import numpy as np
import scipy.sparse as sp
import scs

def min_l1_with_fixed(G=None, Q=None, L=None, c=None, k=None, y=None, A=None, b=None, d=1, verbose=False, params=None, solver=None):
    """
    Solve a problem of the form 
       argmin_u |Gu|_{d,1} + 0.5 u.T Q u + c.T u
           s.t. u[k]= y
                A u = b
    where |x|_{d,1} is the norm which takes the l2 norm of every d adjacent entries before summing them.
    
    e.g. |x|_{1,1} is the familiar l1 norm (the default for this function); |x|_{n,1} is the (unsquared) l2 norm of x.         
        

    Parameters
    ----------
    G : (m*d, n) sparse scipy csr_array, optional
        Matrix to be multiplied inside the norm.  Defaults to sp.eye(n), where n is inferred from another input.
    Q : (n, n) sparse scipy csr_array, optional
        Quadratic form matrix. 0 if not specified.
    L : (?, n) sparse scipy csr_array, optional
        Prefactorization of Q; do not provide Q when providing L.  May make the solve faster when using MOSEK, but slower when using SCS.  
        Q is factored if L is not specified (when using MOSEK).
    c : (n,) numpy array, optional
        Linear term in the objective.  0 if not specified.
    k : (f,) numpy array, optional
        Indices for fixed parts of u.
    y : (f,) numpy array, optional
        Values for fixed parts of u.
    A : (l, n) numpy array, optional
        Linear constraint matrix.
    b : (l, 1) numpy array, optional
        Linear constraint vector.
    d : int, optional
        Number of adjacent vector entries in Gu to take vector norm of before summing.  1 if not specified (corresponds to the standard l1 norm of Gu).
        Solver uses second-order conic constraints for the L1 term if d != 1.
    verbose: string, optional
        Whether to print solver progress and solution status.  False if not specified.
    params: list or dict, optional
        Other arguments provided to the solver, may be useful for e.g. setting tolerances.  
        
        For MOSEK, the input should be a list, where each entry should be a tuple of strings, e.g. ("MSK_DPAR_INTPNT_CO_TOL_REL_GAP", "1.0e-7").
        See https://docs.mosek.com/latest/pythonapi/parameters.html for all of the parameter choices.  Empty list by default.
        
        For SCS, the input should be a dictionary containing parameter names (strings) and values (type may vary).
        See https://www.cvxgrp.org/scs/api/settings.html for all of the parameter choices.  
        The SCS defaults are mostly as in their documentation, except that we use defaults `eps_abs = 1e-9`, `eps_rel = 1e-9`, `eps_infeas = 1e-9`.
    solver: None or string, optional
        Solver type to use; "mosek", "scs", or None (default).  
        
        MOSEK requires users to install the Python module and download a license file; licenses are generally free for academic use and paid for commercial use; see mosek.com for more details.
        SCS is free and exists under an MIT license.
        
        Select "None" (the default) to first try MOSEK, and then fall back to SCS if the license is invalid in some way.
        
    
    Returns
    -------
    u : (n,) numpy array
        Solution to the optimization problem, if optimal.
        
    See Also
    --------
    `min_quad_with_fixed` for solving a quadratic optimization problem while restricting certain degrees of freedom.
    `fixed_dof_solve` for solving a linear system while restricting certain degrees of freedom.
    
    Examples
    --------
    ```python
    >>> import gpytoolbox as gpy
    >>> import numpy as np
    >>> import scipy as sp
    >>> A = np.array([[1, 2]])
    >>> b = np.array([[1]])
    >>> u = gpy.min_l1_with_fixed(A=A, b=b)  # corresponds to argmin |u|_1 s.t. Au == b
    >>> u
    array([-0.        ,  0.5])
    >>> A @ u[:, None]  # check that the linear constraint is satisfied
    array([[1.]])
    ```

    """
    if solver is None:
        # try to use mosek; except any license-related failure and continue with SCS instead
        try:
            globals()["mosek"] = __import__("mosek")
            return min_l1_with_fixed(G, Q, L, c, k, y, A, b, d, verbose, params, solver="mosek")
        except ModuleNotFoundError as e:
            if verbose:
                print("MOSEK Python module not found.")
                print("Continuing with SCS...")
            return min_l1_with_fixed(G, Q, L, c, k, y, A, b, d, verbose, params, solver="scs")
        except mosek.Error as e:
            # all mosek errors related to licensing problems
            license_error_list = {mosek.rescode.err_license, 
                mosek.rescode.err_license_expired,
                mosek.rescode.err_license_version,
                mosek.rescode.err_license_old_server_version,
                mosek.rescode.err_size_license,
                mosek.rescode.err_prob_license,
                mosek.rescode.err_file_license,
                mosek.rescode.err_missing_license_file,
                mosek.rescode.err_size_license_con,
                mosek.rescode.err_size_license_var,
                mosek.rescode.err_size_license_intvar,
                mosek.rescode.err_optimizer_license,
                mosek.rescode.err_flexlm,
                mosek.rescode.err_license_server,
                mosek.rescode.err_license_max,
                mosek.rescode.err_license_moseklm_daemon,
                mosek.rescode.err_license_feature,
                mosek.rescode.err_platform_not_licensed,
                mosek.rescode.err_license_cannot_allocate,
                mosek.rescode.err_license_cannot_connect,
                mosek.rescode.err_license_invalid_hostid,
                mosek.rescode.err_license_server_version,
                mosek.rescode.err_license_no_server_support}
            if e.errno in license_error_list:
                if verbose:
                    print("MOSEK license error: " + str(e.errno))
                    print("Continuing with SCS...")
                return min_l1_with_fixed(G, Q, L, c, k, y, A, b, d, verbose, params, solver="scs")
            else:
                raise e
    elif solver == "mosek":
        globals()["mosek"] = __import__("mosek")
        return _min_l1_with_fixed_mosek(G=G, Q=Q, L=L, c=c, k=k, y=y, A=A, b=b, d=d, verbose=verbose, mosek_params=[] if params is None else params)
    elif solver == "scs":
        scs_params = dict(eps_abs=1e-9, eps_rel=1e-9, eps_infeas=1e-9)
        if not (params is None):
            scs_params.update(params)
        return _min_l1_with_fixed_scs(G=G, Q=Q, L=L, c=c, k=k, y=y, A=A, b=b, d=d, verbose=verbose, scs_params=scs_params)
    else:
        raise ValueError("Solver " + str(solver) + " not supported.")

def _streamprinter(text):
    print(text, flush=True)

def _min_l1_with_fixed_mosek(G=None, Q=None, L=None, c=None, k=None, y=None, A=None, b=None, d=1, verbose=False, mosek_params=[]):
    """Solve the min_l1_with_fixed problem using SCS"""
    
    # reduction to the conic problem
    # min  [u z r].T [c 1 1]
    # s.t. [A   0 0] [u z r].T = [b]
    #      [-G -I 0]          >= [0] } these two only when d=1
    #      [ G -I 0]          >= [0] }
    #      
    #      [G   0 0] [u z r].T  rows \in quadratic cone so that zi >= norm([Gu_i]) only when d>=2
    #      [0   I 0]
    #
    #      [Lt  0 0] [u z r].T + [0]  rows \in a single rotated quadratic cone so that r >= 0.5 u.T Q u = 0.5 |Lu|
    #      [0   0 1]             [0]
    #      [0   0 0]             [1]
    #
    #                       z >= 0
    #                       r >= 0
    
    # infer n from one of the matrices provided
    inferredfrom = ""
    if not (G is None):
        inferredfrom = "G"
        n = G.shape[1]
    elif not (Q is None):
        inferredfrom = "Q"
        n = Q.shape[1]
    elif not (L is None):
        inferredfrom = "L"
        n = L.shape[1]
    elif not (c is None):
        inferredfrom = "c"
        n = c.shape[0]
    elif not (A is None):
        inferredfrom = "A"
        n = A.shape[1]
    elif not (k is None):
        inferredfrom = "k"
        n = np.amax(k)+1
    else:
        raise ValueError("Could not infer the shape of the output vector from input arguments.")
        
    # defaults for some of the inputs if they're None
    if G is None:
        G = sp.eye(n)
    if c is None:
        c = np.zeros(n)
    
    m = G.shape[0] // d # number of subblocks to L1

    # CONSTRUCT VARIABLES CORRESPONDING TO THE GIVEN PROBLEM

    # get cholesky of Q, or transpose L, to get Lt; otherwise set it to None
    Lt, Ltshape = None, 0
    if not (Q is None):
        assert (L is None), "Cannot take both Q and L."
        
        # mosek cholesky decomposition (requires license)
        with mosek.Env() as env:
            Q = sp.csc_matrix(Q)
            # erase all upper triangular elements to feed to mosek
            qi, qj, qv = sp.find(Q)
            lmask = qi>=qj
            Q_tri = sp.csc_matrix((qv[lmask], (qi[lmask], qj[lmask])), shape=Q.shape)
            anzc, aptrc, asubc, avalc = Q_tri.indptr[1:]-Q_tri.indptr[:-1], Q_tri.indptr[Q_tri.indptr!=Q_tri.indptr[-1]], Q_tri.indices, Q_tri.data
            perm, diag, lnzc, lptrc, lensubnval, lsubc, lvalc = env.computesparsecholesky(0, 1, 1e-14, anzc, aptrc, asubc, avalc)
            Lt = sp.csc_array((np.array(lvalc), 
                               np.array(perm)[np.array(lsubc)],
                               np.concatenate([np.array(lptrc), 
                                               np.array([len(lvalc)]*(int(np.amax(np.array(lsubc)+1))-len(lptrc)+1))])), # pad to size of lsubc
                              shape=(n, int(np.amax(np.array(lsubc)+1)))).T
        
        Ltshape = Lt.shape[0]
    if not (L is None):
        Lt = L.T
        Ltshape = Lt.shape[0]
    
    Ashape = 0
    if not (A is None):
        Ashape = A.shape[0]
    
    # convert to csr arrays
    G = sp.csr_array(G)
    if Ashape != 0:
        A = sp.csr_array(A)
    if Ltshape != 0:
        Lt = sp.csr_array(Lt)
    
    # assert shapes
    assert c.shape == (n,), "c shape wrong based on n inferred from " + inferredfrom
    assert G.shape[0]%d == 0, "Row count of G is not a multiple of d"
    assert G.shape == (d*m, n), "G shape wrong based on shape inferred from " + inferredfrom
    assert (y is None and k is None) or not (y is None or k is None), "Only one of k, y provided"
    if not (k is None):
        assert k.shape == y.shape, "k and y must be same shape"
    if not (A is None):
        assert not (b is None), "b must be provided when A is provided"
        assert A.shape[1] == n, "A shape wrong based on n inferred from " + inferredfrom
        assert A.shape[0] == b.shape[0], "A and b have different shapes"
        assert b.shape[1] == 1, "b must have shape (l, 1)"
    if not (b is None):
        assert not (A is None), "A must be provided when b is provided"
    if not (Q is None):
        assert Q.shape == (n, n), "Q shape not (n, n)"
    if not (Lt is None):
        assert Lt.shape[1] == n, "Lt shape not (?, n)"
    
    # pad A to involve the new variables
    moA, mob = None, None
    if d == 1:
        Im = sp.eye(m)
        if Ashape == 0:
            moA = sp.block_array([[-G, -Im],
                                  [ G, -Im]])
            mob = np.zeros((moA.shape[0], 1))
        else:
            moA = sp.block_array([[ A, None],
                                  [-G,  -Im],
                                  [ G,  -Im]])
            mob = np.vstack([b, 
                             np.zeros((2*m, 1))])
        # add a single padding column on the side if there is a quadratic term, for r
        if Ltshape != 0:
            moA = sp.hstack([moA, np.zeros((moA.shape[0], 1))])
    elif d >= 2:
        # If d >= 2, the z and r terms don't give linear constraints.  So just pad with zeros
        if Ashape != 0:
            moA = sp.block_array([[A, sp.csr_array((Ashape, m + (1 if Ltshape != 0 else 0)))]])
            mob = b
    else:
        raise ValueError("d must be at least 1.")
    
    # pad c to involve the new variables
    moc = np.vstack([c[:, None],
                     np.ones((m, 1))])
    if Ltshape != 0:
        moc = np.vstack([moc,
                         1])
    
    # create mosek F, g to involve conic constraints (if necessary)
    moF = None
    if d != 1:
        Im = sp.eye(m)
        if Ltshape != 0:
            moF = sp.block_array([[G, None, None],
                                  [None, Im, None],
                                  [Lt, None, None],
                                  [None, None, sp.csr_array([[1]])],
                                  [None, None, sp.csr_array([[0]])]])
        else:
            moF = sp.block_array([[G,  None],
                                  [None, Im]])
    elif d == 1 and Ltshape != 0:
        moF = sp.block_array([[Lt, sp.csr_array((Ltshape, m)), None],
                              [None, None, sp.csr_array([[1]])],
                              [None, None, sp.csr_array([[0]])]])
    
    
    # CREATE THE TASK AND LOAD IT WITH VARIABLES
    
    with mosek.Task() as task:
        # print to the stream printer if verbose
        if verbose:
            task.set_Stream(mosek.streamtype.log, _streamprinter)
        
        varcount = n + m + (1 if Ltshape != 0 else 0)
        task.appendvars(varcount)
            
        # input c to the task
        for j in range(varcount):
            task.putcj(j, moc[j, 0])
        
        # free variable constraint on the first n variables, otherwise lower bound by zero
        for j in range(varcount):
            if j < n:
                task.putvarbound(j, mosek.boundkey.fr, 0, 0)
            else:
                task.putvarbound(j, mosek.boundkey.lo, 0, 0)
        
        # linear constraints
        if not (moA is None):
            moAi, moAj, moAv = sp.find(moA)
            
            task.appendcons(moA.shape[0])
            task.putaijlist(moAi, moAj, moAv)
            # fixed linear constraint on first Ashape variables, otherwise upper bound by 0
            for j in range(moA.shape[0]):
                if j < Ashape:
                    task.putconbound(j, mosek.boundkey.fx, mob[j, 0], mob[j, 0])
                else:
                    task.putconbound(j, mosek.boundkey.up, 0, mob[j, 0])
        
        # fixed variable constraints based on k, y
        if not (k is None):
            for j in range(k.shape[0]):
                task.putvarbound(k[j], mosek.boundkey.fx, y[j], y[j])
        
        # conic constraints, if necessary
        if not (moF is None):
            moFi, moFj, moFv = sp.find(moF)
            task.appendafes(moF.shape[0])
            task.putafefentrylist(moFi, moFj, moFv)
            # append a single 1 in the last coordinate if there is a quadratic term
            if Ltshape != 0:
                task.putafeg(moF.shape[0]-1, 1)
                # single rotated cone constraint for quadratic optimisation
                # entries 1, r, then (Lt u)_i
                rquadcone = task.appendrquadraticconedomain(Ltshape+2)
                task.appendacc(rquadcone,
                               [moF.shape[0]-1, moF.shape[0]-2] + list(range(moF.shape[0]-Ltshape-2, moF.shape[0]-2)),
                               None) # unused
            
            # append all quadratic cones
            if d != 1:
                for i in range(m):
                    quadcone = task.appendquadraticconedomain(d+1)
                    task.appendacc(quadcone,
                                [d*m+i] + [d*i+j for j in range(d)], # rows from F: zi, Gu_[d adjacent entries corresponding to i]
                                None)
        
        # insert other mosek parameters to the task, if desired
        for a in mosek_params:
            task.putparam(a[0], a[1])
        
        # objective sense
        task.putobjsense(mosek.objsense.minimize)
        
        # optimise
        task.optimize()
        
        # print out a summary of the solution if done
        if verbose:
            task.solutionsummary(mosek.streamtype.msg)
        
        # solution_status
        sol_status = task.getsolsta(mosek.soltype.itr)
        
        if sol_status == mosek.solsta.optimal:
            return np.array(task.getxx(mosek.soltype.itr))[:n] # only send back the first n.  the rest are dummy variables
        elif (sol_status == mosek.solsta.dual_infeas_cer or
              sol_status == mosek.solsta.prim_infeas_cer):
            raise ValueError("Primal or dual infeasibility certificate found.\n")
        elif sol_status == mosek.solsta.unknown:
            raise ValueError("Unknown solution status")
        else:
            raise ValueError("Other solution status")
        
def _min_l1_with_fixed_scs(G=None, Q=None, L=None, c=None, k=None, y=None, A=None, b=None, d=1, verbose=False, scs_params={}):
    """Solve the min_l1_with_fixed problem using SCS"""
    
    # reduction to the conic problem
    # min 0.5 u.T Q u + [c | 1].T [u | z]
    # s.t. [Au   0] [u] + [ s0  ]  = [0]
    #      [kI   0] [z]   [ |   ]    [b]   ;  s0 in the 0 cone
    #
    #      [0  -Im] [u|z].T + [s1]  = [0]     ;  s1 in the positive cone
    #
    #      [G  -Im] [u]  +  [s2]    = [0]      ; s2 in the positive cone (only when d=1)
    #      [-G  Im] [z]     [| ]      [0]
    #
    #      R([-G   0]) [u]   + [s3] = [0]      ; s3 in second order cones (only when d>=2)
    #       ([0   -I]) [z]     [| ]   [0]      ; R reorders entries so that z_i aligns with the d Gu_i
    
    # infer n from one of the matrices provided
    inferredfrom = ""
    if not (G is None):
        inferredfrom = "G"
        n = G.shape[1]
    elif not (Q is None):
        inferredfrom = "Q"
        n = Q.shape[1]
    elif not (L is None):
        inferredfrom = "L"
        n = L.shape[1]
    elif not (c is None):
        inferredfrom = "c"
        n = c.shape[0]
    elif not (A is None):
        inferredfrom = "A"
        n = A.shape[1]
    elif not (k is None):
        inferredfrom = "k"
        n = np.amax(k)+1
    else:
        raise ValueError("Could not infer the shape of the output vector from input arguments.")
    
    # defaults for some of the inputs if they're None
    if G is None:
        G = sp.eye(n)
    if c is None:
        c = np.zeros(n)
    
    m = G.shape[0] // d # number of subblocks to L1
    
    # get Q if provided L
    if not (Q is None):
        assert (L is None), "Cannot take both Q and L."
    if not (L is None):
        Q = L@L.T
    
    # convert to csr arrays
    G = sp.csr_array(G)
    if not (A is None):
        A = sp.csr_array(A)
    
    # assert shapes
    assert c.shape == (n,), "c shape wrong based on n inferred from " + inferredfrom
    assert G.shape[0]%d == 0, "Row count of G is not a multiple of d"
    assert G.shape == (d*m, n), "G shape wrong based on shape inferred from " + inferredfrom
    assert (y is None and k is None) or not (y is None or k is None), "Only one of k, y provided"
    if not (k is None):
        assert k.shape == y.shape, "k and y must be same shape"
    if not (A is None):
        assert not (b is None), "b must be provided when A is provided"
        assert A.shape[1] == n, "A shape wrong based on n inferred from " + inferredfrom
        assert A.shape[0] == b.shape[0], "A and b have different shapes"
        assert b.shape[1] == 1, "b must have shape (l, 1)"
    if not (b is None):
        assert not (A is None), "A must be provided when b is provided"
    if not (Q is None):
        assert Q.shape == (n, n), "Q shape not (n, n)"
    
    # the P matrix given to SCS is the same as our Q matrix.  also pad to include length of z
    scsP=Q
    if not (scsP is None):
        scsP = sp.block_diag([scsP, sp.csr_array((m, m))])
    
    # construct blocks of A
    kI = None
    if not k is None:
        kI = sp.csr_array((np.ones(k.shape[0]), (np.arange(k.shape[0]), k)), shape=(k.shape[0], n))
    Im = sp.eye(m)
    
    scsA = sp.block_array([[A, None],
                           [kI, None],
                           [sp.csr_array((m, n)), -Im]])
    
    if d == 1:
        scsA = sp.vstack([scsA, 
                          sp.block_array([[G,  -Im],
                                          [-G, -Im]])]) 
    elif d >= 2:
        # set up an array which has the entries of each length-(d+1) second-order cone in the order required by SCS
        # i.e. the entries should be in the order z[i], (Gu)[d*i], (Gu)[d*i+1], ... (Gu)[d*i+(d-1)]
        GIm = -sp.block_diag([G, Im])
        scs_Gperm = np.hstack([np.arange(d*m, d*m+m)[:, None], np.reshape(np.arange(d*m), (m, d))]).flatten()
        scs_Gperm_mat = sp.csr_array((np.ones(d*m+m), (np.arange(d*m+m), scs_Gperm)))
        scsA = sp.vstack([scsA,
                          scs_Gperm_mat@GIm])
        
    # construct b
    scsb = np.array([])
    if not (b is None):
        scsb = np.concatenate([scsb, b[:, 0]])
    if not (y is None):
        scsb = np.concatenate([scsb, y])
    scsb = np.concatenate([scsb, np.zeros(scsA.shape[0]-scsb.shape[0])]) # concatenate zeros to match scsA
    
    # get all the correct cone lengths
    ocone_len = (A.shape[0] if not (A is None) else 0) + (k.shape[0] if not (k is None) else 0)
    poscone_len = m + (2*m if d == 1 else 0)
    soc_lens = np.full(m, d+1) if d >= 2 else 0 # m cones of size d+1
    
    # give everything to SCS; convert to csc
    data = dict(A=sp.csc_matrix(scsA),
                b=scsb,
                c=np.concatenate([c, np.ones(m)]))
    if not (scsP is None):
        data |= dict(P=sp.csc_matrix(scsP))
    cone = dict(z=ocone_len,
                l=poscone_len)
    if d >= 2:
        cone |= dict(q=soc_lens)
    
    solver = scs.SCS(data, cone, verbose=verbose, **scs_params)
    
    # solve it
    sol = solver.solve()
    
    # return u if good
    u = sol["x"][:n]
    sol_info = sol["info"]
    
    if verbose:
        print("Solution information:")
        print(sol_info)
    
    # only return if it's an actual solution
    if sol_info["status_val"] == 1:
        return u
    else:
        raise ValueError("Could not find optimal solution, solution exit flag: " + str(sol_info["status_val"]))
