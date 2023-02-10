from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestReadMesh(unittest.TestCase):

    # def test_meshes(self):
    #     meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
    #     for mesh in meshes:
    #         V_py,F_py,UV_py,Ft_py,N_py,Fn_py = \
    #         gpy.read_mesh("test/unit_tests_data/" + mesh,
    #             return_UV=True, return_N=True, reader='Python')

    #         V_cpp,F_cpp,UV_cpp,Ft_cpp,N_cpp,Fn_cpp = \
    #         gpy.read_mesh("test/unit_tests_data/" + mesh,
    #             return_UV=True, return_N=True, reader='C++')

    #         self.assertTrue(np.isclose(V_py,V_cpp).all)
    #         self.assertTrue((F_py==F_cpp).all())
    #         if UV_py is not None:
    #             self.assertTrue(np.isclose(UV_py,UV_cpp).all)
    #         if Ft_py is not None:
    #             self.assertTrue((Ft_py==Ft_cpp).all())
    #         if N_py is not None:
    #             self.assertTrue(np.isclose(N_py,N_cpp).all)
    #         if Fn_py is not None:
    #             self.assertTrue((Fn_py==Fn_cpp).all())

    # def test_stl_reader(self):
    #     stl_meshes = ["sphere_binary.stl", "fox_ascii.stl"]
    #     gt_v_sizes = [4080,1866]
    #     gt_f_sizes = [1360,622]
    #     for mesh in stl_meshes:
    #         V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
    #         self.assertTrue(V.shape[0] == gt_v_sizes[stl_meshes.index(mesh)])
    #         self.assertTrue(F.shape[0] == gt_f_sizes[stl_meshes.index(mesh)])

    def test_ply_reader(self):
        ply_meshes = ["bunny.ply","happy_vrip.ply"]
        for mesh in ply_meshes:
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)


if __name__ == '__main__':
    unittest.main()
