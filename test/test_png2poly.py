import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestPng2Poly(unittest.TestCase):
    def test_simple_pngs(self):
        # Build a polyline; for example, a square
        filename = "test/unit_tests_data/poly.png"
        poly = gpytoolbox.png2poly(filename)
        # There should be two contours: one for each transition
        assert(len(poly)==2)
        # plt.plot(poly[0][:,0],poly[0][:,1])
        # plt.plot(poly[1][:,0],poly[1][:,1])
        # plt.show(block=False)
        # plt.pause(20)
        # plt.close()

        # Test 2: Image from Adobe Illustrator
        filename = "test/unit_tests_data/illustrator.png"
        poly = gpytoolbox.png2poly(filename)
        # There should be four contours: one for each transition in each component
        assert(len(poly)==4)
        # plt.plot(poly[0][:,0],poly[0][:,1])
        # plt.plot(poly[1][:,0],poly[1][:,1])
        # plt.plot(poly[2][:,0],poly[2][:,1])
        # plt.plot(poly[3][:,0],poly[3][:,1])
        # plt.show(block=False)
        # plt.pause(20)
        # plt.close()



if __name__ == '__main__':
    unittest.main()
