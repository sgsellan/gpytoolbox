from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
from gpytoolbox.copyleft import swept_volume
# import polyscope as ps
# import igl

class TestSweptVolume(unittest.TestCase):
    # Would be nice to make this better with more tests, especially groundtruths. Let's improve it once we have more functions (e.g., signed distances).
    # Right now this is just checking that the parameter combinations work
    def test_cube(self):
        np.random.seed(0)
        # Build a cube
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        v = gpytoolbox.normalize_points(v,center=np.array([0.0,0.0,0.0]))
        # Straight line
        transformation_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        transformation_1 = np.array([[1,0,0,0.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        transformations = [transformation_0,transformation_1]
        u,g = swept_volume(v,f,transformations=transformations,eps=0.05,verbose=False)

        # Catmull-Rom without rotations
        translation_0 = np.array([0,0,0])
        translation_1 = np.array([1,0,-1])
        translation_2 = np.array([2,0,1])
        translation_3 = np.array([3,0,0])
        translations = [translation_0,translation_1,translation_2,translation_3]
        u,g = swept_volume(v,f,translations=translations,eps=0.05,verbose=False,align_rotations_with_velocity=False)
        u,g = swept_volume(v,f,translations=translations,eps=0.05,verbose=False,align_rotations_with_velocity=True)
        u,g = swept_volume(v,f,translations=translations,eps=0.05,verbose=False,align_rotations_with_velocity=True,num_faces=200)
        self.assertTrue(np.isclose(g.shape[0]-200,0,atol=3))
    
    def test_with_scale(self):
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        v = gpytoolbox.normalize_points(v,center=np.array([0.0,0.0,0.0]))
        translation_0 = np.array([0,0,0])
        translation_1 = np.array([1,0,-1])
        translation_2 = np.array([2,0,1])
        translation_3 = np.array([3,0,0])
        translations = [translation_0,translation_1,translation_2,translation_3]
        scales = [1,1.5,2,1]
        # scales = [1,1,1,1]
        u,g = swept_volume(v,f,translations=translations,scales=scales,eps=0.05,verbose=False,align_rotations_with_velocity=True)
        # ps.init()
        # ps.register_surface_mesh("test_sv",u,g)
        # ps.show()

if __name__ == '__main__':
    unittest.main()