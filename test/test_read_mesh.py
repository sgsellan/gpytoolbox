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

    def test_stl_reader(self):
        stl_meshes = ["sphere_binary.stl", "fox_ascii.stl"]
        gt_v_sizes = [4080,1866]
        gt_f_sizes = [1360,622]
        for mesh in stl_meshes:
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh,merge_stl=False)
            self.assertTrue(V.shape[0] == gt_v_sizes[stl_meshes.index(mesh)])
            self.assertTrue(F.shape[0] == gt_f_sizes[stl_meshes.index(mesh)])
            self.assertTrue(len(gpy.boundary_vertices(F)) == V.shape[0]) # all vertices are boundary vertices since it is not merged
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh,merge_stl=True)
            # Now the mesh is a single connected mesh, so boundary_vertices will return the correct result
            print(len(gpy.boundary_vertices(F)))
            self.assertTrue(len(gpy.boundary_vertices(F)) == 0)

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
            C = np.random.rand(V.shape[0],4)
            C = np.round(C*255).astype(np.int32)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,C=C,binary=False)
            V_2,F_2,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_C=True)
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue(np.isclose(C_2,C).all)
            self.assertTrue((F_2==F).all())
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            # print(N)
            C = np.random.rand(V.shape[0],4)
            C = np.round(C*255).astype(np.int32)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,C=C,binary=True)
            V_2,F_2,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_C=True)
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue(np.isclose(C_2,C).all)
            self.assertTrue((F_2==F).all())

            # normals and colors
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)

            N = np.random.rand(V.shape[0],3)
            C = np.random.rand(V.shape[0],4)
            C = np.round(C*255).astype(np.uint8)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,N=N,C=C,binary=False)
            V_2,F_2,N_2,_,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_N=True,return_C=True)
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue(np.isclose(N_2,N).all)
            self.assertTrue(np.isclose(C_2,C).all)
            self.assertTrue((F_2==F).all())
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            N = np.random.rand(V.shape[0],3)
            C = np.random.rand(V.shape[0],4)
            C = np.round(C*255).astype(np.uint8)
            gpy.write_mesh("test/unit_tests_data/temp.ply",V,F,N=N,C=C,binary=True)
            V_2,F_2,N_2,_,C_2 = gpy.read_mesh("test/unit_tests_data/temp.ply",return_N=True,return_C=True)
            self.assertTrue(np.isclose(V_2,V).all)
            self.assertTrue(np.isclose(N_2,N).all)
            self.assertTrue(np.isclose(C_2,C).all)
            self.assertTrue((F_2==F).all())


if __name__ == '__main__':
    unittest.main()
