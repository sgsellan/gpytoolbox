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

    def test_stl_read_then_write(self):
        stl_meshes = ["sphere_binary.stl", "fox_ascii.stl"]
        for mesh in stl_meshes:
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            gpy.write_mesh("test/unit_tests_data/temp.stl",V,F,binary=True)
            V_2,F_2 = gpy.read_mesh("test/unit_tests_data/temp.stl")
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue((F_2==F).all())
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            gpy.write_mesh("test/unit_tests_data/temp.stl",V,F,binary=False)
            V_2,F_2 = gpy.read_mesh("test/unit_tests_data/temp.stl")
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue((F_2==F).all())

    def test_ply_read_then_write(self):
        # no normals no colors
        ply_meshes = ["bunny.ply","happy_vrip.ply","example_cube-ascii.ply"]
        for mesh in ply_meshes:
            # no color and no normals
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,binary=True)
            V_2,F_2 = gpy.read_mesh("test/unit_tests_data/temp.ply")
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue((F_2==F).all())
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,binary=False)
            V_2,F_2 = gpy.read_mesh("test/unit_tests_data/temp.ply")
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue((F_2==F).all())
            # normals but no colors
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            # print(N)
            N = np.random.rand(V.shape[0],3)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,N=N,binary=False)
            V_2,F_2,N_2,_ = gpy.read_mesh("test/unit_tests_data/temp.ply",return_N=True)
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue(np.isclose(N_2,N).all)
            self.assertTrue((F_2==F).all())
            # now binary
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            # print(N)
            N = np.random.rand(V.shape[0],3)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,N=N,binary=True)
            V_2,F_2,N_2,_ = gpy.read_mesh("test/unit_tests_data/temp.ply",return_N=True)
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue(np.isclose(N_2,N).all)
            self.assertTrue((F_2==F).all())
            # colors but no normals
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            # print(N)
            for Ctype in ["per_face", "per_vertex"]:
                nc = V.shape[0] if Ctype=="per_vertex" else F.shape[0]
                C = np.random.rand(nc,4)
                C = np.round(C*255).astype(np.int32)
                gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,C=C,binary=False)
                V_2,F_2,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_C=True)
                self.assertTrue(np.isclose(V_2,V).all)
                self.assertTrue(np.isclose(C_2,C).all)
                self.assertTrue((F_2==F).all())
                V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
                # print(N)
                C = np.random.rand(nc,4)
                C = np.round(C*255).astype(np.int32)
                gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,C=C,binary=True)
                V_2,F_2,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_C=True)
                self.assertTrue(np.isclose(V_2,V).all)
                self.assertTrue(np.isclose(C_2,C).all)
                self.assertTrue((F_2==F).all())

                # normals and colors
                V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)

                N = np.random.rand(V.shape[0],3)
                C = np.random.rand(nc,4)
                C = np.round(C*255).astype(np.uint8)
                gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,N=N,C=C,binary=False)
                V_2,F_2,N_2,_,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_N=True,return_C=True)
                self.assertTrue(np.isclose(V_2,V).all)
                self.assertTrue(np.isclose(N_2,N).all)
                self.assertTrue(np.isclose(C_2,C).all)
                self.assertTrue((F_2==F).all())
                V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
                N = np.random.rand(V.shape[0],3)
                C = np.random.rand(nc,4)
                C = np.round(C*255).astype(np.uint8)
                gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,N=N,C=C,binary=True)
                V_2,F_2,N_2,_,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_N=True,return_C=True)
                self.assertTrue(np.isclose(V_2,V).all)
                self.assertTrue(np.isclose(N_2,N).all)
                self.assertTrue(np.isclose(C_2,C).all)
                self.assertTrue((F_2==F).all())


if __name__ == '__main__':
    unittest.main()
