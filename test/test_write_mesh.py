from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import filecmp

class TestWriteMesh(unittest.TestCase):
    def test_obj_read_then_write(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
        writers = ["C++", "Python"]
        for mesh in meshes:
            for writer in writers:
                # Try writing, then reading back.
                V,F,UV,Ft,N,Fn = \
                gpy.read_mesh("test/unit_tests_data/" + mesh,
                    return_UV=True, return_N=True, reader='C++')
                gpy.write_mesh("test/unit_tests_data/temp.obj",V,F,UV,Ft,N,Fn,fmt=None,writer=writer)
                V_2,F_2,UV_2,Ft_2,N_2,Fn_2 = \
                gpy.read_mesh("test/unit_tests_data/temp.obj",
                    return_UV=True, return_N=True, reader='C++')

                self.assertTrue(np.isclose(V_2,V).all)
                self.assertTrue((F_2==F).all())
                if UV_2 is not None:
                    self.assertTrue(np.isclose(UV_2,UV).all)
                if Ft_2 is not None:
                    self.assertTrue((Ft_2==Ft).all())
                if N_2 is not None:
                    self.assertTrue(np.isclose(N_2,N).all)
                if Fn_2 is not None:
                    self.assertTrue((Fn_2==Fn).all())

    def test_obj_default_Ft_Fn(self):
        meshes = ["cone.obj", "mountain.obj", "hemisphere.obj"]
        writers = ["C++", "Python"]
        for mesh in meshes:
            for writer in writers:
                V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
                dummy_UV = V[:,:2]
                dummy_N = gpy.per_vertex_normals(V,F)

                gpy.write_mesh("test/unit_tests_data/temp.UV.N.0.obj",V,F,dummy_UV,F,dummy_N,F,writer=writer)
                gpy.write_mesh("test/unit_tests_data/temp.UV.N.1.obj",V,F,UV=dummy_UV,N=dummy_N,writer=writer)
                gpy.write_mesh("test/unit_tests_data/temp.UV.0.obj",V,F,UV=dummy_UV,Ft=F,writer=writer)
                gpy.write_mesh("test/unit_tests_data/temp.UV.1.obj",V,F,UV=dummy_UV,writer=writer)
                gpy.write_mesh("test/unit_tests_data/temp.N.0.obj",V,F,N=dummy_N,Fn=F,writer=writer)
                gpy.write_mesh("test/unit_tests_data/temp.N.1.obj",V,F,N=dummy_N,writer=writer)

                self.assertTrue(filecmp.cmp("test/unit_tests_data/temp.UV.N.0.obj", "test/unit_tests_data/temp.UV.N.1.obj", shallow=False))
                self.assertTrue(filecmp.cmp("test/unit_tests_data/temp.UV.0.obj", "test/unit_tests_data/temp.UV.1.obj", shallow=False))
                self.assertTrue(filecmp.cmp("test/unit_tests_data/temp.N.0.obj", "test/unit_tests_data/temp.N.1.obj", shallow=False))

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
