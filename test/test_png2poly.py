import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestPng2Poly(unittest.TestCase):
    def test_simple_pngs(self):
        # Build a polyline
        filename = "test/unit_tests_data/poly.png"
        poly = gpytoolbox.png2poly(filename)
        # There should be two contours: one for each transition
        self.assertTrue(len(poly)==2)
        # plt.plot(poly[0][:,0],poly[0][:,1])
        # plt.plot(poly[1][:,0],poly[1][:,1])
        # plt.show(block=False)
        # plt.pause(20)
        # plt.close()

        # Test 2: Image from Adobe Illustrator
        filename = "test/unit_tests_data/illustrator.png"
        poly = gpytoolbox.png2poly(filename)
        # There should be four contours: one for each transition in each component
        self.assertTrue(len(poly)==4)

    def test_rotation(self):
        poly = gpytoolbox.png2poly("test/unit_tests_data/rectangle.png")
        self.assertTrue(len(poly)==1)
        V = poly[0]
        # X dimension should be bigger:
        vmin = np.amin(V,axis=0)
        vmax = np.amax(V,axis=0)
        xlength = vmax[0] - vmin[0]
        ylength = vmax[1] - vmin[1]
        # This used to fail. It shouldn't now:
        self.assertTrue(xlength>ylength)
        



if __name__ == '__main__':
    unittest.main()
