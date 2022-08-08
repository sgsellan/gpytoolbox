import numpy as np
import scipy as sp


def fixed_dof_solve(A, b=None, k=None, y=None):
    """Solves a linear system while fixing certain degrees of freedom for which
    the linear system is to be ignored during solution.
    
    For the linear system `A*u = b`, the linear system will be enforced at all
    the rows that don't correspond to fixed degrees of freedom, and the fixed
    degrees of freedom will be enforced on their respective rows.

    This can be used to implement finite differences with Dirichlet boundary
    conditions by fixing the boundary degrees of freedom to the appropriate
    boundary values.
    
    Parameters
    ----------
    A : (n,n) scipy csc matrix
        square matrix for the linear system
    b : (n,) or (n,p) numpy float array
        right-hand side of the linear system
    k : (o,) numpy int array
        index vector of fixed degrees of freedom
    y : (o,) or (o,p) numpy float array
        what the degrees of freedom are fixed to, `u[k] == y` or `u[k,:] == y`


    Returns
    -------
    u : (n,) or (n,p) numpy float array such that
        `A[not k,:] * u == b[not k]` or `A[not k,:] * u == b[not k,:]`
        and `u[k] == y` or `u[k,:] == y`.


    See Also
    --------
    min_quad_with_fixed


    Examples
    --------
    TODO
    """

    return fixed_dof_solve_precompute(A, k).solve(b, y)


# This is written in snake_case on purpose, so constructing the class looks just
# like calling a function.
class fixed_dof_solve_precompute:
    def __init__(self, A, k=None):
        """Prepare a precomputation object to efficiently solve a linear system
        while fixing certain degrees of freedom for which the linear system is to be
        ignored during solution.
        
        For the linear system `A*u = b`, the linear system will be enforced at all
        the rows that don't correspond to fixed degrees of freedom, and the fixed
        degrees of freedom will be enforced on their respective rows.

        This can be used to implement finite differences with Dirichlet boundary
        conditions by fixing the boundary degrees of freedom to the appropriate
        boundary values.
        
        Parameters
        ----------
        A : (n,n) scipy csc matrix
            square matrix for the linear system
        k : (o,) numpy int array
            index vector of fixed degrees of freedom


        Returns
        -------
        precomputed : precomputation object that can be used to solve the problem



        See Also
        --------
        min_quad_with_fixed


        Examples
        --------
        TODO
        """

        self.n = A.shape[0]
        assert A.shape[1] == self.n
        assert self.n>0
        # self.A = A.copy()

        if k is None:
            self.o = 0
            self.k = None
        else:
            self.o = k.shape[0]
            assert k.shape == (self.o,)
            assert np.min(k)>=0 and np.max(k)<self.n
            assert np.unique(k).shape == k.shape, "No duplicate indices"
            self.k = k.copy()

        # If k is provided, remove these degrees of freedom.
        if self.o==0:
            self.ki = None
            self.n_reduced = self.n
            self.Ared = A
            self.A_for_extra_b = None
        else:
            self.ki = np.setdiff1d(np.arange(0, self.n), self.k)
            self.n_reduced = self.ki.shape[0]
            assert self.n_reduced == self.n - self.k.shape[0]
            self.Ared = A[self.ki, :][:, self.ki]
            self.A_for_extra_b = A[self.ki, :][:, self.k]

        splu = sp.sparse.linalg.splu(self.Ared)
        self.solver = lambda x : splu.solve(x)
    def solve(self, b=None, y=None):
        """Solve the prefactored linear system
        while fixing certain degrees of freedom for which the linear system is to be
        ignored during solution.
        
        For the linear system `A*u = b`, the linear system will be enforced at all
        the rows that don't correspond to fixed degrees of freedom, and the fixed
        degrees of freedom will be enforced on their respective rows.

        This can be used to implement finite differences with Dirichlet boundary
        conditions by fixing the boundary degrees of freedom to the appropriate
        boundary values.
        
        Parameters
        ----------
        b : (n,) or (n,p) numpy float array
            right-hand side of the linear system
        y : (o,) or (o,p) numpy float array
            what the degrees of freedom are fixed to, `u[k] == y` or `u[k,:] == y`


        Returns
        -------
        u : (n,) or (n,p) numpy float array such that
            `A[not k,:] * u == b[not k]` or `A[not k,:] * u == b[not k,:]`
            and `u[k] == y` or `u[k,:] == y`.


        See Also
        --------
        min_quad_with_fixed


        Examples
        --------
        TODO
        """

        def cp(x):
            if x is None or np.isscalar(x):
                return 0
            if len(x.shape)==1:
                return 0
            return x.shape[1]
        p = max([cp(b), cp(y)])

        assert b is None or np.isscalar(b) or (p==0 and b.shape==(self.n,)) or (p>0 and b.shape==(self.n,p))
        assert y is None or (np.isscalar(y) and self.o>0) or (p==0 and y.shape==(self.o,)) or (p>0 and y.shape==(self.o,p))

        # Get everything to full dimensions
        if b is None:
            b = 0.
        if np.isscalar(b):
            if p==0:
                b = np.full(self.n, b)
            else:
                b = np.full((self.n,p), b)
        if y is None and self.o>0:
            y = 0.
        if np.isscalar(y) and self.o>0:
            if p==0:
                y = np.full(self.o, y)
            else:
                y = np.full((self.o,p), y)

        # Modified rhs based on known values
        if self.o==0:
            rhs = b
        else:
            b_reduced = b[self.ki] if p==0 else b[self.ki,:]
            rhs = b_reduced - self.A_for_extra_b*y
        ured = self.solver(rhs)

        if self.o==0:
            u = ured
        else:
            if p==0:
                u = np.empty(self.n, dtype=np.float64)
                u[self.ki] = ured
                u[self.k] = y
            else:
                u = np.empty((self.n,p), dtype=np.float64)
                u[self.ki,:] = ured
                u[self.k,:] = y

        return u


