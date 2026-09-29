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
    
    def test_quad_obj(self):
        # Pure quad and mixed triangle/quad OBJ meshes. The Python and C++
        # readers must agree exactly, including texture and normal indices.
        meshes = ["quad_cube.obj", "quad_cube_uv_n.obj", "mixed_tri_quad.obj"]
        for mesh in meshes:
            V_py,F_py,UV_py,Ft_py,N_py,Fn_py = \
            gpy.read_mesh("test/unit_tests_data/" + mesh,
                return_UV=True, return_N=True, reader='Python')
            V_cpp,F_cpp,UV_cpp,Ft_cpp,N_cpp,Fn_cpp = \
            gpy.read_mesh("test/unit_tests_data/" + mesh,
                return_UV=True, return_N=True, reader='C++')
            V_def,F_def = gpy.read_mesh("test/unit_tests_data/" + mesh)

            self.assertEqual(F_py.shape[1], 4)
            self.assertTrue(np.array_equal(V_py,V_cpp))
            self.assertTrue(np.array_equal(F_py,F_cpp))
            self.assertTrue(np.array_equal(Ft_py,Ft_cpp))
            self.assertTrue(np.array_equal(Fn_py,Fn_cpp))
            self.assertTrue(np.array_equal(V_py,V_def))
            self.assertTrue(np.array_equal(F_py,F_def))
            if UV_py is None:
                self.assertEqual(UV_cpp.size, 0)
            else:
                self.assertTrue(np.array_equal(UV_py,UV_cpp))
            if N_py is None:
                self.assertEqual(N_cpp.size, 0)
            else:
                self.assertTrue(np.array_equal(N_py,N_cpp))

        # Check the quad cube with texture coordinates and normals against
        # ground truth.
        V,F,UV,Ft,N,Fn = gpy.read_mesh("test/unit_tests_data/quad_cube_uv_n.obj",
            return_UV=True, return_N=True)
        self.assertEqual(V.shape, (8,3))
        self.assertEqual(UV.shape, (4,2))
        self.assertEqual(N.shape, (6,3))
        self.assertTrue(np.array_equal(F, np.array([[0,3,2,1],[4,5,6,7],
            [0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])))
        self.assertTrue(np.array_equal(Ft, np.tile([0,1,2,3], (6,1))))
        self.assertTrue(np.array_equal(Fn, np.repeat(np.arange(6)[:,None], 4, axis=1)))

        # Check the mixed triangle/quad mesh against ground truth: triangles
        # are padded with -1 in F, Ft and Fn.
        for reader in ["Python", "C++"]:
            V,F,UV,Ft,N,Fn = gpy.read_mesh("test/unit_tests_data/mixed_tri_quad.obj",
                return_UV=True, return_N=True, reader=reader)
            self.assertEqual(V.shape, (9,3))
            self.assertEqual(UV.shape, (5,2))
            self.assertEqual(N.shape, (9,3))
            self.assertTrue(np.array_equal(F, np.array([[4,5,8,-1],
                [0,3,2,1],[0,1,5,4],[5,6,8,-1],[1,2,6,5],[2,3,7,6],
                [3,0,4,7],[6,7,8,-1],[7,4,8,-1]])))
            tri = F[:,3] == -1
            self.assertTrue(np.array_equal(Ft[tri], np.tile([0,1,4,-1], (4,1))))
            self.assertTrue(np.array_equal(Ft[~tri], np.tile([0,1,2,3], (5,1))))
            self.assertTrue(np.array_equal(Fn, np.array([[5,5,5,-1],
                [0,0,0,0],[1,1,1,1],[6,6,6,-1],[2,2,2,2],[3,3,3,3],
                [4,4,4,4],[7,7,7,-1],[8,8,8,-1]])))

    def test_non_tri_quad_obj(self):
        # Faces with more than four vertices are still not supported.
        with open("test/unit_tests_data/temp.pentagon.obj", "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv -1 0.5 0\n"
                    "f 1 2 3\nf 1 2 3 4 5\n")
        for reader in ["Python", "C++"]:
            with self.assertRaises(Exception):
                gpy.read_mesh("test/unit_tests_data/temp.pentagon.obj",
                    reader=reader)

    def test_ply_index_vs_indices_faces(self):
        # this used to fail:
        V, F = gpy.read_mesh("test/unit_tests_data/mesh-indices.ply")
        # this used to return an empty F, but not anymore:
        self.assertTrue(F.shape[0] == 934)


if __name__ == '__main__':
    unittest.main()
