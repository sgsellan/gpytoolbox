from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
from .context import unittest

class TestMinL1WithFixed(unittest.TestCase):
    # the following only test MOSEK if an activated license is registered on the computer; otherwise, the MOSEK tests instead default to SCS
    # MOSEK: equality tolerances are set to (absolute) 1e-5 by default
    # SCS: equality tolerances are set to (absolute) 1e-8 by default
    
    def test_simple_example(self):
        # test that it works for d=1 with a single example using a "ground truth" numerical solution
        G = np.diag([1, 2, 3])
        U = np.array([[1, 7, 5],
                      [7, 2, 0.1],
                      [5, 0.1, 3]])
        Q = U.T @ U
        c = np.array([10.123, 2.456, 3.678])
        A = np.array([[7.5, 5.2, 1.234]])
        b = np.array([[1]])
        
        u_star = np.array([ 0.02202419, 0.20346248, -0.18086411]) # numerical solution
        ml1wf_mosek = gpy.min_l1_with_fixed(G=G, Q=Q, c=c, A=A, b=b, d=1, verbose=False, solver=None)
        ml1wf_scs = gpy.min_l1_with_fixed(G=G, Q=Q, c=c, A=A, b=b, d=1, verbose=False, solver="scs")
        
        self.assertTrue(np.isclose(u_star, ml1wf_mosek, atol=1e-9).all())
        self.assertTrue(np.isclose(u_star, ml1wf_scs, atol=1e-9).all())
    
    def test_aligns_with_random_min_quad_with_fixed(self):
        # test that the function aligns with min_quad_with_fixed for random sizes and a quarter of the values fixed
        # scs seems to sometimes fail on QPs.  if it fails, restarting it often works.
        # it is nondeterministic, and its seed cannot be set in the api; the failure is also computer-dependent.
        # typically it needs to retry at most two or three times over all of the runs (empirically) (see prints)
        
        rng = np.random.default_rng(0)
        for size in range(4, 100):
            L = rng.random((size, size))
            L = np.linalg.qr(L)[0] # L orthonormal
            D = np.diag(np.linspace(1e-12, 1, num=size)) # spectrum increases linearly
            Q = sp.sparse.csc_matrix(L.T @ D @ L) # pos semidef
            c = rng.random(size)
            tofix = int(0.25*size)
            k = rng.choice(size, tofix, replace=False)
            y = rng.random(tofix)
            
            # min quad with fixed solution
            mqwf = gpy.min_quad_with_fixed(Q=Q, c=c, k=k, y=y)
            
            # min l1 with fixed solution and check
            try:
                ml1wf_mosek = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=1, verbose=False, solver=None)
                ml1wf_scs = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=1, verbose=False, solver="scs")
            except ValueError:
                print("Size " + str(size) + " solver failure (d=1); trying again...")
                ml1wf_mosek = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=1, verbose=False, solver=None)
                ml1wf_scs = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=1, verbose=False, solver="scs")
            self.assertTrue(np.isclose(mqwf, ml1wf_mosek, rtol=0, atol=1e-5).all())
            self.assertTrue(np.isclose(mqwf, ml1wf_scs, rtol=0, atol=1e-8).all())
        
            # also do min l1 with fixed with d=2 check; only when size is even
            if size % 2 == 0:
                try:
                    ml1wfd_mosek = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=2, verbose=False, solver=None)
                    ml1wfd_scs = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=2, verbose=False, solver="scs")
                except ValueError:
                    print("Size " + str(size) + " solver failure (d=2); trying again...")
                    ml1wfd_mosek = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=2, verbose=False, solver=None)
                    ml1wfd_scs = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=2, verbose=False, solver="scs")
                self.assertTrue(np.isclose(mqwf, ml1wfd_mosek, rtol=0, atol=1e-5).all())
                self.assertTrue(np.isclose(mqwf, ml1wfd_scs, rtol=0, atol=1e-8).all())
                
            # also do min l1 with fixed with d=2 check; only when size is divisible by 3
            if size % 3 == 0:
                try:
                    ml1wfd_mosek = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=3, verbose=False, solver=None)
                    ml1wfd_scs = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=3, verbose=False, solver="scs")
                except ValueError:
                    print("Size " + str(size) + " solver failure (d=3); trying again...")
                    ml1wfd_mosek = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=3, verbose=False, solver=None)
                    ml1wfd_scs = gpy.min_l1_with_fixed(G=sp.sparse.csr_array(Q.shape), Q=Q, c=c, k=k, y=y, d=3, verbose=False, solver="scs")
                self.assertTrue(np.isclose(mqwf, ml1wfd_mosek, rtol=0, atol=1e-5).all())
                self.assertTrue(np.isclose(mqwf, ml1wfd_scs, rtol=0, atol=1e-8).all())
                
    def test_l1_unconstrained_reconstruction(self):
        # test that the function correctly reconstructs a vector if the only constraint is L1 distance to that vector
        rng = np.random.default_rng(1)
        for size in range(4, 100):
            In = sp.sparse.eye(size)
            G = sp.sparse.hstack([In, -In])
            u0 = rng.random(size)
            k = np.array(range(size, 2*size))
            ml1wf_mosek = gpy.min_l1_with_fixed(G=G, k=k, y=u0, solver=None)
            ml1wf_scs = gpy.min_l1_with_fixed(G=G, k=k, y=u0, solver="scs")
            
            self.assertTrue(np.isclose(u0, ml1wf_mosek[:size], rtol=0, atol=1e-5).all())
            self.assertTrue(np.isclose(u0, ml1wf_scs[:size], rtol=0, atol=1e-8).all())
    
    def test_l1_d_unconstrained_reconstruction(self):
        # test that the function correctly reconstructs a set of vectors if the only constraints are l1 distance to those vectors
        # test for vectors of size between 2 and 10, and vector counts between 4 and 100
        rng = np.random.default_rng(2)
        for dim in range(2, 10):
            for veccount in range(4, 100):
                In = sp.sparse.eye(dim*veccount)
                G = sp.sparse.hstack([In, -In])
                u0 = rng.random(dim*veccount)
                k = np.array(range(dim*veccount, 2*dim*veccount))
                ml1wf_mosek = gpy.min_l1_with_fixed(G=G, k=k, y=u0, d=dim, solver=None)
                ml1wf_scs = gpy.min_l1_with_fixed(G=G, k=k, y=u0, d=dim, solver="scs")
                
                self.assertTrue(np.isclose(u0, ml1wf_mosek[:dim*veccount], rtol=0, atol=1e-5).all())
                self.assertTrue(np.isclose(u0, ml1wf_scs[:dim*veccount], rtol=0, atol=1e-8).all())
                
    def test_l1_2d_opt(self):
        # test that for a lot of random line constraints in 2d, the result matches
        # the analytical solution (the line's intersection with the x or y axes that lies closest to the origin)
        rng = np.random.default_rng(3)
        for i in range(100):
            angle = 2*np.pi*rng.random()
            randvec = np.array([[np.cos(angle)], 
                                [np.sin(angle)]])
            A = randvec.T
            b = np.array([[1]])
            
            ml1wf_mosek = gpy.min_l1_with_fixed(G=np.eye(2), A=A, b=b, d=1, verbose=False, solver=None)
            ml1wf_scs = gpy.min_l1_with_fixed(G=np.eye(2), A=A, b=b, d=1, verbose=False, solver="scs")
            
            # true solution:
            if np.abs(A[0, 0]) > np.abs(A[0, 1]):
                gt = np.array([1.0/A[0, 0], 0])
            else:
                gt = np.array([0, 1.0/A[0, 1]])
            
            self.assertTrue(np.isclose(ml1wf_mosek, gt, rtol=0, atol=1e-5).all()) # solver gives exact solution in 2d
            self.assertTrue(np.isclose(ml1wf_scs, gt, rtol=0, atol=1e-8).all())
    
    def test_l1_d_sphere_opt(self):
        # use a unit vector to define a hyperplane in any dimension; then find the vector with the smallest norm
        # within that hyperplane; it should be the same as the original unit vector, by construction
        rng = np.random.default_rng(4)
        for dim in range(2, 10):
            for i in range(100):
                # create somewhat-random (random in each coordinate, normalized) vectors on the unit sphere
                randvec = rng.random(dim)
                randvec = randvec/np.linalg.norm(randvec)
                
                # create a hyperplane at distance 1 from the origin and constrain; then, assert that the solutions are the same as the original vectors
                A = randvec[None, :]
                b = np.array([[1]])
                
                ml1wf_mosek = gpy.min_l1_with_fixed(G=None, A=A, b=b, d=dim, verbose=False, solver=None)
                ml1wf_scs = gpy.min_l1_with_fixed(G=None, A=A, b=b, d=dim, verbose=False, solver="scs")
                
            
                # the solution should be exactly the randomly chosen vector
                self.assertTrue(np.isclose(ml1wf_mosek, randvec, rtol=0, atol=1e-5).all())
                self.assertTrue(np.isclose(ml1wf_scs, randvec, rtol=0, atol=1e-9).all())

    def test_l1_errors_on_bad_input(self):
        # test a small number of good / bad inputs to make sure they do / don't raise errors
        
        G = np.eye(3)
        A = np.eye(3)
        b = np.ones((3, 1))
        k = np.array([0])
        y = np.array([1])
        
        # bad solver name
        with self.assertRaises(ValueError):
            gpy.min_l1_with_fixed(G=G, solver="moscs")
            
        # k but no y; y but no k
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, k=k, solver=None)
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, k=k, solver="scs")
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, y=y, solver=None)
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, y=y, solver="scs")
            
        # A but no b; b but no A
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, A=A, solver=None)
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, A=A, solver="scs")
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, b=b, solver=None)
        with self.assertRaises(AssertionError):
            gpy.min_l1_with_fixed(G=G, b=b, solver="scs")
        
        # test that params doesn't break SCS
        gpy.min_l1_with_fixed(G=G, params={"adaptive_scale":True,
                                           "time_limit_secs":0}, solver="scs")

if __name__=="__main__":
    unittest.main()
