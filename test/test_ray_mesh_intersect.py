from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestRayMeshIntersect(unittest.TestCase):
    def test_simple_cube(self):
        # This is a cube, centered at the origin, with side length 1
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        cam_pos = np.array([[1,0.1,0.1],[1,0.2,0.0]])
        cam_dir = np.array([[-1,0,0],[-1,0,0]])
        t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,cam_dir,v,f)
        # There should only be two hits: let's check the output
        self.assertTrue(t.shape[0]==2)
        self.assertTrue(np.isclose(t[0],0.5))
        groundtruth_intersection = np.array([[0.5,0.1,0.1],[0.5,0.2,0.0]])
        intersection = cam_pos + t[:,None]*cam_dir
        self.assertTrue(np.isclose(groundtruth_intersection,intersection).all())
        t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,cam_dir,v,f,use_embree=False)
        # There should only be two hits: let's check the output
        self.assertTrue(t.shape[0]==2)
        self.assertTrue(np.isclose(t[0],0.5))
        groundtruth_intersection = np.array([[0.5,0.1,0.1],[0.5,0.2,0.0]])
        intersection = cam_pos + t[:,None]*cam_dir
        self.assertTrue(np.isclose(groundtruth_intersection,intersection).all())
    def test_if_no_hit(self):
        # Purposefully creating a situation where the ray doesn't hit
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        cam_pos = np.array([[2,2.0,0.1],[0,2.2,2.0]])
        cam_dir = np.array([[0,0,-1],[1,0,0]])
        t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,cam_dir,v,f)
        self.assertTrue((t==np.Inf).all())
        self.assertTrue((ids==-1).all())
        self.assertTrue((l==0.0).all())
        t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,cam_dir,v,f,use_embree=False)
        self.assertTrue((t==np.Inf).all())
        self.assertTrue((ids==-1).all())
        self.assertTrue((l==0.0).all())
    def test_embree_vs_no_embree(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "bunny.obj", "mountain.obj"]
        num_samples = 100 # Should be more but this is already pretty slow
        for mesh in meshes:
            v,f = gpytoolbox.read_mesh("test/unit_tests_data/" + mesh)
            v = gpytoolbox.normalize_points(v)
            v,f,_,_ = gpytoolbox.decimate(v,f,face_ratio=0.1)
            # print(f.shape[0])
            # Generate random point
            cam_pos = np.random.rand(num_samples,3)-1
            # cam_dir = 2*np.random.rand(num_samples,3)-4
            te, idse, le = gpytoolbox.ray_mesh_intersect(cam_pos,-cam_pos,v,f,use_embree=True)
            t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,-cam_pos,v,f,use_embree=False)
            # To compare
            t[t==np.Inf] = 0.
            te[te==np.Inf] = 0.
            # print(t)
            # print(te)
            self.assertTrue(np.isclose(te-t,0,atol=1e-4).all())
            self.assertTrue(np.isclose(ids-idse,0,atol=1e-4).all())
            self.assertTrue(np.isclose(l-le,0,atol=1e-4).all())
            # Now precomputing tree:
            C,W,CH,_,_,tri_ind = gpytoolbox.initialize_aabbtree(v,F=f)
            t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,-cam_pos,v,f,use_embree=False,C=C,CH=CH,W=W,tri_ind=tri_ind)
            self.assertTrue(np.isclose(te-t,0,atol=1e-4).all())
            self.assertTrue(np.isclose(ids-idse,0,atol=1e-4).all())
            self.assertTrue(np.isclose(l-le,0,atol=1e-4).all())




if __name__ == '__main__':
    unittest.main()