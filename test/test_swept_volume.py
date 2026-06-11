from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
from gpytoolbox.copyleft import swept_volume
from gpytoolbox.copyleft.swept_volume import velocity_alignment_rotations
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

    def test_velocity_alignment_on_spiral(self):
        # Regression test for issue #136: on a spiral path, aligning rotations
        # with velocity used to introduce a spurious roll about the velocity
        # axis (computed each keyframe from a fixed global reference), making
        # the shape flip on one side of the helix. The rotation-minimizing
        # frame should instead produce proper rotations whose local x-axis
        # tracks the velocity and which vary smoothly (no sudden flips).
        radius, start_y, end_y, num_steps, pitch = 0.025, -0.07, 0.07, 50, 0.2
        h_step = (end_y - start_y) / num_steps
        translations = []
        for i in range(num_steps):
            theta = 2 * np.pi * i / (num_steps * pitch)
            translations.append(np.array([radius * np.cos(theta),
                                          start_y + i * h_step,
                                          radius * np.sin(theta)]))

        rotations = velocity_alignment_rotations(translations)
        self.assertEqual(len(rotations), num_steps)

        velocities = []
        for i in range(num_steps):
            if i == 0:
                vel = translations[1] - translations[0]
            elif i == num_steps - 1:
                vel = translations[-1] - translations[-2]
            else:
                vel = translations[i + 1] - translations[i - 1]
            velocities.append(vel / np.linalg.norm(vel))

        x_axis = np.array([1.0, 0.0, 0.0])
        for R, vel in zip(rotations, velocities):
            # proper rotation
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-8))
            self.assertTrue(np.isclose(np.linalg.det(R), 1.0, atol=1e-8))
            # local x-axis aligns with the velocity
            self.assertTrue(np.allclose(R @ x_axis, vel, atol=1e-7))

        # frame must vary smoothly: no near-180-degree flip between neighbours
        up = np.array([0.0, 1.0, 0.0])
        max_up_jump = max(np.linalg.norm(rotations[i] @ up - rotations[i - 1] @ up)
                          for i in range(1, num_steps))
        self.assertLess(max_up_jump, 1.0)

if __name__ == '__main__':
    unittest.main()
    
