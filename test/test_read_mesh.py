from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestReadMesh(unittest.TestCase):

    def test_meshes(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
        for mesh in meshes:
            V_py,F_py,UV_py,Ft_py,N_py,Fn_py = \
            gpy.read_mesh("test/unit_tests_data/" + mesh,
                return_UV=True, return_N=True, reader='Python')

            V_cpp,F_cpp,UV_cpp,Ft_cpp,N_cpp,Fn_cpp = \
            gpy.read_mesh("test/unit_tests_data/" + mesh,
                return_UV=True, return_N=True, reader='C++')

            self.assertTrue(np.isclose(V_py,V_cpp).all)
            self.assertTrue((F_py==F_cpp).all())
            if UV_py is not None:
                self.assertTrue(np.isclose(UV_py,UV_cpp).all)
            if Ft_py is not None:
                self.assertTrue((Ft_py==Ft_cpp).all())
            if N_py is not None:
                self.assertTrue(np.isclose(N_py,N_cpp).all)
            if Fn_py is not None:
                self.assertTrue((Fn_py==Fn_cpp).all())

if __name__ == '__main__':
    unittest.main()
