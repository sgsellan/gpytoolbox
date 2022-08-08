from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import scipy as sp

class TestMinQuadWithFixed(unittest.TestCase):

    def test_unconstrained(self):
        Adense = np.array([[1.,5.,0.3],[2.,3.,7.],[-4.,3.5,-1.]])
        A = sp.sparse.csc_matrix(Adense)

        k = None
        y = None

        self.solution_verifier(A,None,k,y,np.array([0.,0.,0.]))
        self.solution_verifier(A,1.,k,y,np.array([-0.08333333,  0.21212121,  0.07575758]))
        self.solution_verifier(A,np.array([0.4,0.7,-0.9]),k,y,np.array([0.24673913, 0.02964427, 0.01679842]))
        self.solution_verifier(A,np.array([[0.4,3.],[0.7,-0.5],[-0.9,2.]]),k,y,np.array([[ 0.24673913,  0.11413043],[ 0.02964427,  0.59881423],[ 0.01679842, -0.36067194]]))


    def test_constrained_norhs(self):
        Adense = np.array([[1.,5.,0.3,0.5],[2.,3.,7.,-4.],[-4.,3.5,-1.,1.2],[-2.4,-0.7,1.8,0.3]])
        A = sp.sparse.csc_matrix(Adense)

        b = None
        k = np.array([0,2])

        self.solution_verifier(A,b,k,None,np.array([0.,0.,0.,0.]))
        self.solution_verifier(A,b,k,2.6,np.array([2.6       , 0.41052632, 2.6       , 6.15789474]))
        self.solution_verifier(A,b,k,np.array([-0.2,0.6]),np.array([-0.2       ,  3.88421053,  0.6       ,  3.86315789]))
        self.solution_verifier(A,b,k,np.array([[-0.2,0.7],[0.6,-2.]]),np.array([[ -0.2       ,   0.7       ],[  3.88421053, -13.10526316],[  0.6       ,  -2.        ],[  3.86315789, -12.97894737]]))


    def test_constrained_rhs(self):
        Adense = np.array([[1.,5.,0.3,0.5],[2.,3.,7.,-4.],[-4.,3.5,-1.,1.2],[-2.4,-0.7,1.8,0.3]])
        A = sp.sparse.csc_matrix(Adense)

        k = np.array([0,2])

        self.solution_verifier(A,None,k,None,np.array([0.,0.,0.,0.]))
        self.solution_verifier(A,2.,k,None,np.array([ 0.        , -4.52631579,  0.        , -3.89473684]))
        self.solution_verifier(A,None,k,-1.,np.array([-1.        , -0.15789474, -1.        , -2.36842105]))
        self.solution_verifier(A,np.array([-0.9,3.,1.7,0.3]),k,np.array([-0.2,0.6]),np.array([-0.2       ,  2.77894737,  0.6       ,  2.28421053]))
        self.solution_verifier(A,np.array([[-0.9,2.1],[3.,-0.7],[1.7,2.3],[0.3,4.]]),k,np.array([[-0.2,0.7],[0.6,-2.]]),np.array([[ -0.2       ,   0.7       ],[  2.77894737, -21.41578947],[  0.6       ,  -2.        ],[  2.28421053, -19.03684211]]))


    def test_random(self):
        return None
        rng = np.random.default_rng()

        for i in range(20):
            n = rng.integers(10,50)
            p = rng.integers(0,5)

            Adense = rng.random((n,n))
            if np.linalg.cond(Adense) > 1. / np.finfo(Adense.dtype).eps:
                continue
            A = sp.sparse.csc_matrix(Adense)

            if rng.random() < 0.5:
                if p==0:
                    b = rng.random(n)
                else:
                    b = rng.random((n,p))
            else:
                if rng.random() < 0.5:
                    b = rng.random()
                else:
                    b = None

            if rng.random() < 0.5:
                k = np.unique(rng.integers(0,n,rng.integers(1,n//2)))
                o = k.shape[0]
                if rng.random() < 0.5:
                    y = rng.random()
                else:
                    if p==0:
                        y = rng.random(o)
                    else:
                        y = rng.random((o,p))
            else:
                k = None
                y = None

        self.solution_verifier(A, b, k, y)


    def solution_verifier(self, A, b=None, k=None, y=None, ugt=None):
        u = gpy.fixed_dof_solve(A,b,k,y)
        u2 = gpy.fixed_dof_solve_precompute(A,k).solve(b,y)
        self.assertTrue(np.isclose(u,u2).all())

        if ugt is not None:
            self.assertTrue(np.isclose(u,ugt).all())

        if b is None:
            b = np.zeros(A.shape[0]) if y is None or np.isscalar(y) or len(y.shape)==1 else np.zeros((A.shape[0],y.shape[1]))
        if k is None:
            self.assertTrue(np.isclose(A*u,b).all())
        else:
            notk = np.setdiff1d(np.arange(0, A.shape[0]), k)
            if np.isscalar(b):
                self.assertTrue(np.isclose(A[notk,:]*u,b).all())
            else:
                self.assertTrue(np.isclose(A[notk,:]*u,b[notk]).all())
            if y is None:
                y = 0.
            self.assertTrue((u[k]==y).all())

if __name__ == '__main__':
    unittest.main()