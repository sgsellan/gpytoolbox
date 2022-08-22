from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestCatmullRomSpline(unittest.TestCase):
    def test_simple_curve(self):
        # Generate many examples
        np.random.seed(0)
        for size in range(4,100):
            P = np.random.rand(size,2)
            rand_time =  0.99*np.random.rand(1)
            rand_point = gpytoolbox.catmull_rom_spline(rand_time,P)
            dist = 10
            # Test continuity
            for exp in range(10,15):
                rand_time_forward = rand_time + (2**(-exp))
                current_point = gpytoolbox.catmull_rom_spline(rand_time_forward,P)
                self.assertTrue(dist>np.linalg.norm(current_point-rand_point))
                dist = np.linalg.norm(current_point-rand_point)
                # print(dist)
            # Test derivatives
            # All the keyframe times
            time_keyframes = np.linspace(0,1,P.shape[0])
            tau = time_keyframes[1]
            for bb in range(0,P.shape[0]):
                if bb==0:
                    true_derivative = (P[bb+1,:] - P[bb,:])/(tau)
                    sample_points = gpytoolbox.catmull_rom_spline(np.array([time_keyframes[bb]+1e-8,0.0]),P)
                    fd_derivative = (sample_points[0,:] - sample_points[1,:])/(1e-8)
                    # print(true_derivative)
                    # print(fd_derivative)
                    self.assertTrue(np.isclose(true_derivative-fd_derivative,0.0,atol=1e-4).all())
                elif bb==(P.shape[0]-1):
                    true_derivative = (P[bb,:] - P[bb-1,:])/(tau)
                    sample_points = gpytoolbox.catmull_rom_spline(np.array([1.0,time_keyframes[bb]-1e-8]),P)
                    fd_derivative = (sample_points[0,:] - sample_points[1,:])/(1e-8)
                    # print(true_derivative)
                    # print(fd_derivative)
                    self.assertTrue(np.isclose(true_derivative-fd_derivative,0.0,atol=1e-4).all())
                else:
                    true_derivative = (P[bb+1,:] - P[bb-1,:])/(2*tau)
                    sample_points = gpytoolbox.catmull_rom_spline(np.array([time_keyframes[bb]+1e-8,time_keyframes[bb]-1e-8]),P)
                    fd_derivative = (sample_points[0,:] - sample_points[1,:])/(2*1e-8)
                    # print(true_derivative)
                    # print(fd_derivative)
                    self.assertTrue(np.isclose(true_derivative-fd_derivative,0.0,atol=1e-4).all())






# if __name__ == '__main__':
#     unittest.main()