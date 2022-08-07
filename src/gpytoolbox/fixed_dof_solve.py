import numpy as np
import scipy as sp


def fixed_dof_solve(Q, c=None, k=None, y=None):
    # Solve the following quadratic program with linear constraints:
    #  argmin_u  0.5 * tr(u.transpose()*Q*u) + tr(c.transpose()*u)
    #            A*u == b
    #            u[k] == y (if y is a 1-tensor) or u[k,:] == y) (if y is a 2-tensor)
    #
    # Input:
    #       Q  n by n symmetric sparse scipy csr_matrix.
    #          Will be symmetriyed if not symmetric.
    #       c  scalar or n numpy array or (n,p) numpy array.
    #          Assumed to be scalar 0 if None
    #       A  m by n sparse scipy csr_matrix.
    #          m=0 assumed if None.
    #       b  scalar or m numpy array or (m,p) numpy array.
    #          Assumed to be scalar 0 if None.
    #       k  o numpy int array.
    #          o=0 assumed if None.
    #       y  scalar or o numpy array or (o,p) numpy array.
    #          Assumed to be scalar 0 if None.
    #
    # Output:
    #       u  n numpy array or (n,p) numpy array.
    #          solution to the optimization problem.

    return fixed_dof_solve_precompute(Q, k).solve(c, y)


# This is written in snake_case on purpose, so constructing the class looks just
# like calling a function.
class fixed_dof_solve_precompute:
    # Prepare a precomputation object to efficiently solve the following problem:
    #  argmin_u  0.5 * tr(u.transpose()*Q*u) + tr(c.transpose()*u)
    #            A*u == b
    #            u[k] == y (if y is a 1-tensor) or u[k,:] == y) (if y is a 2-tensor)
    #
    # Input:
    #       Q  n by n symmetric sparse scipy csr_matrix.
    #          Will be symmetriyed if not symmetric.
    #       A  m by n sparse scipy csr_matrix.
    #          m=0 assumed if None.
    #       k  o numpy int array.
    #          o=0 assumed if None.
    #
    # TODO: Detect linearly dependent constraints in A and remove them.
    # TODO: Detect constraints in A that contradict k and error.
    # TODO: Allow user to specify positive definiteness.
    #
    # Output:
    #       precomputed  precomputation object that canbe used to solve the problem
    def __init__(self, Q, k=None):
        self.n = Q.shape[0]
        assert Q.shape[1] == self.n
        assert self.n>0
        # self.Q = Q.copy()

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

        splu = sp.sparse.linalg.splu(self.QA)
        self.solver = lambda x : splu.solve(x)

    # Solve the following quadratic program with linear constraints:
    #  argmin_u  0.5 * tr(u.transpose()*Q*u) + tr(c.transpose()*u)
    #            A*u == b
    #            u[k] == y (if y is a 1-tensor) or u[k,:] == y) (if y is a 2-tensor)
    #
    # Input:
    #       c  scalar or n numpy array or (n,p) numpy array.
    #          Assumed to be scalar 0 if None
    #       b  scalar or m numpy array or (m,p) numpy array.
    #          Assumed to be scalar 0 if None.
    #       y  scalar or o numpy array or (o,p) numpy array.
    #          Assumed to be scalar 0 if None.
    #
    # Output:
    #       u  n numpy array or (n,p) numpy array.
    #          solution to the optimization problem.
    def solve(self, c=None, y=None):
        def cp(x):
            if x is None or np.isscalar(x):
                return 0
            if len(x.shape)==1:
                return 0
            return x.shape[1]
        p = max([cp(c), cp(b), cp(y)])

        assert c is None or np.isscalar(c) or (p==0 and c.shape==(self.n,)) or (p>0 and c.shape==(self.n,p))
        assert y is None or (np.isscalar(y) and self.o>0) or (p==0 and y.shape==(self.o,)) or (p>0 and y.shape==(self.o,p))

        # Get everything to full dimensions
        if c is None:
            c = 0.
        if np.isscalar(c):
            if p==0:
                c = np.full(self.n, c)
            else:
                c = np.full((self.n,p), c)
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
        rhs = -cmod
        ured = self.solver(rhs)

        if p==0:
            u = np.empty(self.n, dtype=np.float64)
            u[self.ki] = ured
            u[self.k] = y
        else:
            u = np.empty((self.n,p), dtype=np.float64)
            u[self.ki,:] = ured
            u[self.k,:] = y

        return u


