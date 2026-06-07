from .context import numpy as np
from .context import unittest
from .context import gpytoolbox as gpy
import time

class TestParticleSwarm(unittest.TestCase):
    def test_1d_swarm(self):
        # Random value between -10 and 10
        for i in range(1):
            # seed
            np.random.seed(i)
            val = np.random.rand()*20-10
            def func(x):
                return (x-val)**2
            lb = np.array([-10])
            ub = np.array([10])
            x,f = gpy.particle_swarm(func,lb,ub,verbose=False,max_iter=1000)
            # print(val)
            # print(x)
            self.assertTrue(np.isclose(x,val,atol=1e-3).all())
    def test_1d_swarm_ring(self):
        # Random value between -10 and 10
        for i in range(1):
            # seed
            np.random.seed(i)
            val = np.random.rand()*20-10
            def func(x):
                return (x-val)**2
            lb = np.array([-10])
            ub = np.array([10])
            x,f = gpy.particle_swarm(func,lb,ub,verbose=False,max_iter=1000, topology='ring')
            # print(val)
            # print(x)
            self.assertTrue(np.isclose(x,val,atol=1e-3).all())
    def test_2d_swarm(self):
        # Random value between -10 and 10
        for i in range(1):
            # seed
            np.random.seed(i)
            val = np.random.rand(2)*20-10
            def func(x):
                return np.sum((x-val)**2)
            lb = np.array([-10,-10])
            ub = np.array([10,10])
            x,f = gpy.particle_swarm(func,lb,ub,verbose=False,max_iter=1000)
            self.assertTrue(np.isclose(x,val,atol=1e-3).all())

    def test_2d_dropwave(self):
        for i in range(1):
            # seed
            np.random.seed(i)
            random_center = np.random.rand(2)*2-1
            def dropwave_function(x):
                x = x - random_center
                return -(1 + np.cos(12*np.sqrt(np.sum(x**2))))/(0.5*np.sum(x**2) + 2)
            lb = np.array([-5,-5])
            ub = np.array([5,5])
            x,f = gpy.particle_swarm(dropwave_function,lb,ub,verbose=False,max_iter=1000)
            # print(x)
            self.assertTrue(np.isclose(x,random_center,atol=1e-3).all())

    def test_2d_dropwave_ring(self):
        for i in range(1):
            # seed
            np.random.seed(i)
            random_center = np.random.rand(2)*2-1
            def dropwave_function(x):
                x = x - random_center
                return -(1 + np.cos(12*np.sqrt(np.sum(x**2))))/(0.5*np.sum(x**2) + 2)
            lb = np.array([-5,-5])
            ub = np.array([5,5])
            x,f = gpy.particle_swarm(dropwave_function,lb,ub,verbose=False,max_iter=100,topology='full')
            xring,fring = gpy.particle_swarm(dropwave_function,lb,ub,verbose=False,max_iter=100,topology='ring')
            # print(x)
            self.assertTrue(np.isclose(x,random_center,atol=1e-3).all())


    def test_cpp_vs_python_speed(self):
        # Confirms the C++ binding is actually being triggered by comparing
        # wall-clock time of the two backends on the same problem.
        try:
            from gpytoolbox_bindings import _particle_swarm_cpp_impl  # noqa: F401
        except ImportError:
            self.skipTest("C++ binding not available")

        np.random.seed(0)
        random_center = np.random.rand(2)*2-1
        def dropwave_function(x):
            x = x - random_center
            return -(1 + np.cos(12*np.sqrt(np.sum(x**2))))/(0.5*np.sum(x**2) + 2)
        lb = np.array([-5,-5])
        ub = np.array([5,5])

        n_particles = 200
        max_iter = 500

        np.random.seed(0)
        t0 = time.perf_counter()
        x_cpp, f_cpp = gpy.particle_swarm(dropwave_function, lb, ub,
                                          n_particles=n_particles,
                                          max_iter=max_iter,
                                          verbose=False, use_cpp=True)
        t_cpp = time.perf_counter() - t0

        np.random.seed(0)
        t0 = time.perf_counter()
        x_py, f_py = gpy.particle_swarm(dropwave_function, lb, ub,
                                        n_particles=n_particles,
                                        max_iter=max_iter,
                                        verbose=False, use_cpp=False)
        t_py = time.perf_counter() - t0

        speedup = t_py / t_cpp if t_cpp > 0 else float('inf')
        print(
            "\n[particle_swarm timing] n_particles={}, max_iter={}\n"
            "  C++:    {:.4f} s  (f = {:.6f})\n"
            "  Python: {:.4f} s  (f = {:.6f})\n"
            "  speedup: {:.2f}x".format(
                n_particles, max_iter, t_cpp, f_cpp, t_py, f_py, speedup))

        # Both should converge to roughly the same minimum.
        self.assertTrue(np.isclose(x_cpp, random_center, atol=1e-2).all())
        self.assertTrue(np.isclose(x_py, random_center, atol=1e-2).all())
        # And the C++ path should be faster — that's the whole point.
        self.assertLess(t_cpp, t_py)


if __name__ == '__main__':
    unittest.main()
