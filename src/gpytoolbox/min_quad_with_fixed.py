import numpy as np
import scipy as sp


def min_quad_with_fixed(Q,
    c=None,
    A=None,
    b=None,
    k=None,
    y=None):
    """Solve the following quadratic program with linear constraints:
    ```
    argmin_u  0.5 * tr(u.transpose()*Q*u) + tr(c.transpose()*u)
        A*u == b
        u[k] == y (if y is a 1-tensor) or u[k,:] == y) (if y is a 2-tensor)
    ```

    Parameters
    ----------
    Q : (n,n) symmetric sparse scipy csr_matrix
        This matrix will be symmetrized if not exactly symmetric.
    c : None or scalar or (n,) numpy array or (n,p) numpy array
        Assumed to be scalar 0 if None
    A : None or (m,n) sparse scipy csr_matrix
        m=0 assumed if None
    b : None or scalar or (m,) numpy array or (m,p) numpy array
        Assumed to be scalar 0 if None
    k : None or (o,) numpy array
        o=0 assumed if None
    y : None or scalar or (o,) numpy array or (o,p) numpy array
        Assumed to be scalar 0 if None

    Returns
    -------
    u : (n,) numpy array or (n,p) numpy array
        Solution to the optimization problem

    Examples
    --------
    TODO
    
    """

    return min_quad_with_fixed_precompute(Q, A, k).solve(c, b, y)


# This is written in snake_case on purpose, so constructing the class looks just
# like calling a function.
class min_quad_with_fixed_precompute:

    def __init__(self,
        Q,
        A=None,
        k=None):
        """Prepare a precomputation object to efficiently solve the following 
        constrained optimization problem:
        ```
        argmin_u  0.5 * tr(u.transpose()*Q*u) + tr(c.transpose()*u)
            A*u == b
            u[k] == y (if y is a 1-tensor) or u[k,:] == y) (if y is a 2-tensor)
        ```

        Parameters
        ----------
        Q : (n,n) symmetric sparse scipy csr_matrix
            This matrix will be symmetrized if not exactly symmetric.
        A : None or (m,n) sparse scipy csr_matrix
            m=0 assumed if None
        k : None or (o,) numpy array
            o=0 assumed if None

        Returns
        -------
        precomputed : instance of class min_quad_with_fixed_precompute
            precomputation object that can be used to solve the optimization problem

        Examples
        --------
        TODO
        
        """

        self.n = Q.shape[0]
        assert Q.shape[1] == self.n
        assert self.n>0
        # self.Q = Q.copy()

        if A is None:
            self.m = 0
            self.A = None
        else:
            self.m = A.shape[0]
            assert A.shape[1] == self.n
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

        # If Q is not exactly symmetric, symmetrize it.
        # Equality test from https://stackoverflow.com/a/30685839
        if (Q != Q.transpose()).nnz != 0:
            Q = 0.5 * (Q + Q.transpose())

        # If k is provided, remove these degrees of freedom.
        # These two are maps to go between full DOF to reduced DOF
        # Instead of 0.5 * u.transpose()*Q*u + c.transpose()*u, after this
        # reduction we have 0.5 * u[ki].transpose()*Qred*u[ki]
        #                     + (c+ Q_for_extra_c*u[k]).transpose() * u[ki]
        if self.o==0:
            self.ki = None
            self.n_reduced = self.n
            Qred = Q
            self.Q_for_extra_c = None
        else:
            self.ki = np.setdiff1d(np.arange(0, self.n), self.k)
            self.n_reduced = self.ki.shape[0]
            assert self.n_reduced == self.n - self.k.shape[0]
            Qred = Q[self.ki, :][:, self.ki]
            self.Q_for_extra_c = Q[self.ki, :][:, self.k]


        if self.m==0:
            # The system will be positive semidefinite if Q is, use Cholmod.
            # CSC is more efficient per the documentation.
            self.QA = sp.sparse.csc_matrix(Qred)
            try:
                import sksparse
                self.solver = sksparse.cholmod.cholesky(self.QA)
            except:
                splu = sp.sparse.linalg.splu(self.QA)
                self.solver = lambda x : splu.solve(x)

        else:
            # If A*u == b, but parts of u are fixed, we need to account for
            # this: Amod*u[ku] == b - A_for_extra_b*u[k]
            if self.o==0:
                Amod = A
                self.A_for_extra_b = None
            else:
                Amod = A[:, self.ki]
                self.A_for_extra_b = A[:, self.k]
            # If A is provided, the linear system we end up solving is
            # [Q, A.transpose(); A, 0] == [c; b] (+ accommodating k)
            # CSC is more efficient per the documentation.
            self.QA = sp.sparse.bmat([[Qred, Amod.transpose()], [Amod, None]],
                format='csc')
            # If A is provided, the system will be indefinite, use SuperLU.
            splu = sp.sparse.linalg.splu(self.QA)
            self.solver = lambda x : splu.solve(x)


    def solve(self,
        c=None,
        b=None,
        y=None):
        """Solve the following quadratic program with linear constraints:
        ```
        argmin_u  0.5 * tr(u.transpose()*Q*u) + tr(c.transpose()*u)
            A*u == b
            u[k] == y (if y is a 1-tensor) or u[k,:] == y) (if y is a 2-tensor)
        ```

        Parameters
        ----------
        c : None or scalar or (n,) numpy array or (n,p) numpy array
            Assumed to be scalar 0 if None
        b : None or scalar or (m,) numpy array or (m,p) numpy array
            Assumed to be scalar 0 if None
        y : None or scalar or (o,) numpy array or (o,p) numpy array
            Assumed to be scalar 0 if None

        Returns
        -------
        u : (n,) numpy array or (n,p) numpy array
            Solution to the optimization problem

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
        p = max([cp(c), cp(b), cp(y)])

        assert c is None or np.isscalar(c) or (p==0 and c.shape==(self.n,)) or (p>0 and c.shape==(self.n,p))
        assert b is None or (np.isscalar(b) and self.m>0) or (p==0 and b.shape==(self.m,)) or (p>0 and b.shape==(self.m,p))
        assert y is None or (np.isscalar(y) and self.o>0) or (p==0 and y.shape==(self.o,)) or (p>0 and y.shape==(self.o,p))

        # Get everything to full dimensions
        if c is None:
            c = 0.
        if np.isscalar(c):
            if p==0:
                c = np.full(self.n, c)
            else:
                c = np.full((self.n,p), c)
        if b is None and self.m>0:
            b = 0.
        if np.isscalar(b) and self.m>0:
            if p==0:
                b = np.full(self.m, b)
            else:
                b = np.full((self.m,p), b)
        if y is None and self.o>0:
            y = 0.
        if np.isscalar(y) and self.o>0:
            if p==0:
                y = np.full(self.o, y)
            else:
                y = np.full((self.o,p), y)

        # Modified rhs based on known values
        if self.o==0:
            cmod = c
        else:
            c_reduced = c[self.ki] if p==0 else c[self.ki,:]
            cmod = c_reduced + self.Q_for_extra_c*y

        # We need to solve [Q, A.transpose(); A, 0] = [cmod; b] with the
        # reduced matrix Q.
        if self.m==0:
            rhs = -cmod
        else:
            # Modify b based on known values
            bmod = b if self.o==0 else (b - self.A_for_extra_b*y)
            rhs = np.concatenate([-cmod,bmod], axis=0)
        ured = self.solver(rhs)

        # Discard the dummy degrees of freedom from A*u == b
        ured = ured[0:self.n_reduced] if p==0 else ured[0:self.n_reduced,:]

        # Undo the u[k]==y reduction
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


