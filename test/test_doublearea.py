from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestDoublearea(unittest.TestCase):
    def test_polyline_lengths(self):
        # This is a cube, centered at the origin, with side length 1
        # v,f = gpy.read_mesh("test/unit_tests_data/cube.obj")
        #
        # Let's make up a simple polyline
        v = np.array([[0],[0.2],[0.5],[0.98],[1.0]])
        A = gpytoolbox.doublearea(v)
        true_lengths = np.array([0.2,0.3,0.48,0.02])
        self.assertTrue(np.isclose(A-2.0*true_lengths,0.0).all())

    def test_single_triangle_2d(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        A = gpytoolbox.doublearea(v,f)
        self.assertTrue(np.isclose(A - 2.0*0.5,0.0).all())
        # print(G-igl.grad(np.hstack((v,np.zeros((v.shape[0],1)))),f))
    
    def test_single_triangle_3d(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        A = gpytoolbox.doublearea(v,f)
        self.assertTrue(np.isclose(A - 2.0*0.5,0.0).all())
        # print(G-igl.grad(np.hstack((v,np.zeros((v.shape[0],1)))),f))

    def test_2d_regular(self):
        v,f = gpytoolbox.regular_square_mesh(40)
        A = gpytoolbox.doublearea(v,f)
        # Side should be 2/39 
        # So area is 
        gt_dblarea = (2/39)**2.0
        self.assertTrue(np.isclose(A - gt_dblarea,0.0).all())
        # Check should be unsigned
        f_backwards = f[:,[0,2,1]]
        A = gpytoolbox.doublearea(v,f_backwards)
        # Check should be unsigned
        self.assertTrue(np.isclose(A - gt_dblarea,0.0).all())
        # Now do signed
        A = gpytoolbox.doublearea(v,f_backwards,signed=True)
        # Check should be unsigned
        self.assertTrue(np.isclose(A + gt_dblarea,0.0).all())


    def test_orientation(self):
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        A = gpytoolbox.doublearea(v,f)
        self.assertTrue((A>=0).all())
        f_backwards = f[:,[0,2,1]]
        A = gpytoolbox.doublearea(v,f_backwards)
        # Check should be unsigned
        self.assertTrue((A>=0).all())

        


if __name__ == '__main__':
    unittest.main()