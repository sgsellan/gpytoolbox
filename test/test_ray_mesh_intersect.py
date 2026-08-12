from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import gpytoolbox_bindings
import time
from unittest import mock

_HAS_EMBREE = getattr(gpytoolbox_bindings, "_has_embree", True)

class TestRayMeshIntersect(unittest.TestCase):
    def test_default_falls_back_without_embree(self):
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        cam_pos = np.array([[1, 0.1, 0.1], [1, 0.2, 0.0]])
        cam_dir = np.array([[-1, 0, 0], [-1, 0, 0]])
        with mock.patch.object(gpytoolbox_bindings, "_has_embree", False):
            fallback = gpytoolbox.ray_mesh_intersect(cam_pos, cam_dir, v, f)
        portable = gpytoolbox.ray_mesh_intersect(
            cam_pos, cam_dir, v, f, use_embree=False)

        for actual, expected in zip(fallback, portable):
            np.testing.assert_allclose(actual, expected)

    def test_precompute_falls_back_without_embree(self):
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        cam_pos = np.array([[1, 0.1, 0.1], [1, 0.2, 0.0]])
        cam_dir = np.array([[-1, 0, 0], [-1, 0, 0]])
        with mock.patch.object(gpytoolbox_bindings, "_has_embree", False):
            intersector = gpytoolbox.ray_mesh_intersect_precompute(v, f)
            actual = gpytoolbox.ray_mesh_intersect(
                cam_pos, cam_dir, v, f, intersector=intersector)
        expected = gpytoolbox.ray_mesh_intersect(
            cam_pos, cam_dir, v, f, use_embree=False)

        for cached, uncached in zip(actual, expected):
            np.testing.assert_allclose(cached, uncached)

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
        self.assertTrue((t==np.inf).all())
        self.assertTrue((ids==-1).all())
        self.assertTrue((l==0.0).all())
        t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,cam_dir,v,f,use_embree=False)
        self.assertTrue((t==np.inf).all())
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
            t[t==np.inf] = 0.
            te[te==np.inf] = 0.
            # print(t)
            # print(te)
            self.assertTrue(np.isclose(te-t,0,atol=1e-4).all())
            self.assertTrue(np.isclose(ids-idse,0,atol=1e-4).all())
            self.assertTrue(np.isclose(l-le,0,atol=1e-4).all())
            # Now precomputing tree:
            C,W,CH,_,_,tri_ind,_ = gpytoolbox.initialize_aabbtree(v,F=f)
            t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos,-cam_pos,v,f,use_embree=False,C=C,CH=CH,W=W,tri_ind=tri_ind)
            self.assertTrue(np.isclose(te-t,0,atol=1e-4).all())
            self.assertTrue(np.isclose(ids-idse,0,atol=1e-4).all())
            self.assertTrue(np.isclose(l-le,0,atol=1e-4).all())




    @unittest.skipUnless(_HAS_EMBREE, "requires an Embree-enabled build")
    def test_intersector_cache_matches_uncached(self):
        # The cached ray_mesh_intersect_precompute must produce identical results to the
        # uncached path.
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        rng = np.random.default_rng(0)
        n_rays = 500
        origins = rng.uniform(-2, 2, size=(n_rays, 3))
        dirs = rng.standard_normal(size=(n_rays, 3))

        t_u, id_u, l_u = gpytoolbox.ray_mesh_intersect(origins, dirs, v, f)

        rmi = gpytoolbox.ray_mesh_intersect_precompute(v, f)
        t_c, id_c, l_c = gpytoolbox.ray_mesh_intersect(
            origins, dirs, v, f, intersector=rmi)

        # Replace inf with 0 (as the existing tests do) so comparison is safe.
        t_u_finite = np.where(np.isfinite(t_u), t_u, 0.0)
        t_c_finite = np.where(np.isfinite(t_c), t_c, 0.0)
        self.assertTrue(np.allclose(t_u_finite, t_c_finite, atol=1e-5))
        self.assertTrue(np.array_equal(id_u, id_c))
        self.assertTrue(np.allclose(l_u, l_c, atol=1e-5))

        # And the intersector's direct .intersect call is consistent too.
        t_d, id_d, l_d = rmi.intersect(origins, dirs)
        self.assertTrue(np.array_equal(id_d, id_c))

    @unittest.skipUnless(_HAS_EMBREE, "requires an Embree-enabled build")
    def test_intersector_cache_is_faster(self):
        # Sanity check the whole point of the cache: many calls against the
        # same mesh are faster with the precomputed Embree scene than without.
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        rng = np.random.default_rng(0)
        n_rays = 200
        n_iters = 20

        # Warm-up (Embree scene init can hit caches, ensure both paths get one).
        gpytoolbox.ray_mesh_intersect(
            rng.uniform(-2, 2, size=(n_rays, 3)),
            rng.standard_normal(size=(n_rays, 3)),
            v, f)

        # Uncached: rebuilds the Embree scene on every call.
        t0 = time.perf_counter()
        for _ in range(n_iters):
            origins = rng.uniform(-2, 2, size=(n_rays, 3))
            dirs = rng.standard_normal(size=(n_rays, 3))
            gpytoolbox.ray_mesh_intersect(origins, dirs, v, f)
        t_uncached = time.perf_counter() - t0

        # Cached: builds once, reuses.
        rmi = gpytoolbox.ray_mesh_intersect_precompute(v, f)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            origins = rng.uniform(-2, 2, size=(n_rays, 3))
            dirs = rng.standard_normal(size=(n_rays, 3))
            gpytoolbox.ray_mesh_intersect(
                origins, dirs, v, f, intersector=rmi)
        t_cached = time.perf_counter() - t0

        speedup = t_uncached / t_cached if t_cached > 0 else float('inf')
        print(
            "\n[ray_mesh_intersect cache] n_rays={}, n_iters={}, "
            "bunny ({} verts, {} tris)\n"
            "  uncached: {:.4f} s\n"
            "  cached:   {:.4f} s\n"
            "  speedup:  {:.2f}x".format(
                n_rays, n_iters, v.shape[0], f.shape[0],
                t_uncached, t_cached, speedup))
        self.assertLess(t_cached, t_uncached)


if __name__ == '__main__':
    unittest.main()
