from .context import numpy as np
from .context import unittest
from .context import gpytoolbox as gpy

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


if __name__ == '__main__':
    unittest.main()
