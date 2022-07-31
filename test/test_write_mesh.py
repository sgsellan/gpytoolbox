from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestWriteMesh(unittest.TestCase):

    def test_read_then_write(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
        for mesh in meshes:
            V_cpp,F_cpp,UV_cpp,Ft_cpp,N_cpp,Fn_cpp = \
            gpy.read_mesh("test/unit_tests_data/" + mesh,
                return_UV=True, return_N=True, reader='C++')
            gpy.write_mesh("test/unit_tests_data/temp.obj",V_cpp,F_cpp,UV_cpp,Ft_cpp,N_cpp,Fn_cpp,fmt=None,writer='C++')
            V_cpp_2,F_cpp_2,UV_cpp_2,Ft_cpp_2,N_cpp_2,Fn_cpp_2 = \
            gpy.read_mesh("test/unit_tests_data/temp.obj",
                return_UV=True, return_N=True, reader='C++')

            self.assertTrue(np.isclose(V_cpp_2,V_cpp).all)
            self.assertTrue((F_cpp_2==F_cpp).all())
            if UV_cpp_2 is not None:
                self.assertTrue(np.isclose(UV_cpp_2,UV_cpp).all)
            if Ft_cpp_2 is not None:
                self.assertTrue((Ft_cpp_2==Ft_cpp).all())
            if N_cpp_2 is not None:
                self.assertTrue(np.isclose(N_cpp_2,N_cpp).all)
            if Fn_cpp_2 is not None:
                self.assertTrue((Fn_cpp_2==Fn_cpp).all())

if __name__ == '__main__':
    unittest.main()
