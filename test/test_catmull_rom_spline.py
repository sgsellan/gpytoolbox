from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
from gpytoolbox.copyleft import swept_volume
import matplotlib.pyplot as plt
# import polyscope as ps
# import igl

class TestCatmullRomSpline(unittest.TestCase):
    def test_simple_curve(self):
        P = np.random.rand(4,2)
        T = np.linspace(0,1,100)
        PT = gpytoolbox.catmull_rom_spline(T,P)
        plt.plot(PT[:,0],PT[:,1])
        plt.show()


# if __name__ == '__main__':
#     unittest.main()