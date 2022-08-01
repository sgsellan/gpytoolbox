from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


class TestSubdivide(unittest.TestCase):

    def test_polyline(self):
        v = np.array([[0],[0.2],[0.5],[0.98],[1.0]])
        f = gpy.edge_indices(v.shape[0])
        vu,fu = gpy.subdivide(v,f)
        vu_gt = np.array([[0.  ],
       [0.2 ],
       [0.5 ],
       [0.98],
       [1.  ],
       [0.1 ],
       [0.35],
       [0.74],
       [0.99]])
        fu_gt = np.array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [5, 1],
       [6, 2],
       [7, 3],
       [8, 4]])
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())

        self.consistency(v,f)


    def test_single_triangle_2d(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        vu,fu = gpy.subdivide(v,f)
        vu_gt = np.array([[0. , 0. ],
       [1. , 0. ],
       [0. , 1. ],
       [0.5, 0. ],
       [0. , 0.5],
       [0.5, 0.5]])
        fu_gt = np.array([[0, 3, 4],
       [3, 1, 5],
       [4, 5, 2],
       [4, 3, 5]])
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())

        vu,fu = gpy.subdivide(v,f,method='loop')
        vu_gt = np.array([[0.125, 0.125],
       [0.75 , 0.125],
       [0.125  , 0.75 ],
       [0.5  , 0.   ],
       [0.   , 0.5  ],
       [0.5  , 0.5  ]])
        fu_gt = np.array([[0, 3, 4],
       [3, 1, 5],
       [4, 5, 2],
       [4, 3, 5]])
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())

        self.consistency(v,f)
    

    def test_single_triangle_3d(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
        f = np.array([[0,1,2]],dtype=int)
        vu,fu = gpy.subdivide(v,f)
        vu_gt = np.array([[0. , 0. , 0.],
       [1. , 0. , 0.],
       [0. , 1. , 0.],
       [0.5, 0. , 0.],
       [0. , 0.5, 0.],
       [0.5, 0.5, 0.]])
        fu_gt = np.array([[0, 3, 4],
       [3, 1, 5],
       [4, 5, 2],
       [4, 3, 5]])
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())

        vu,fu = gpy.subdivide(v,f,method='loop')
        vu_gt = np.array([[0.125, 0.125, 0.],
       [0.75 , 0.125, 0.],
       [0.125   , 0.75 , 0.],
       [0.5  , 0.   , 0.],
       [0.   , 0.5  , 0.],
       [0.5  , 0.5  , 0.]])
        fu_gt = np.array([[0, 3, 4],
       [3, 1, 5],
       [4, 5, 2],
       [4, 3, 5]])
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())

        self.consistency(v,f)


    def test_meshes(self):
        meshes = ["armadillo.obj", "bunny_oded.obj", "bunny.obj", "mountain.obj"]
        for mesh in meshes:
            v,f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            self.consistency(v,f)


    def test_bunny_oded(self):
        #Ground-truth test a mesh without boundary
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        vu_gt,fu_gt = gpy.read_mesh("test/unit_tests_data/upsampled_bunny_oded.obj")
        vu,fu = gpy.subdivide(v,f)
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())


        vu_gt,fu_gt = gpy.read_mesh("test/unit_tests_data/looped_bunny_oded.obj")
        vu,fu = gpy.subdivide(v,f,method='loop')
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())


    def test_mountain(self):
        #Ground-truth test a mesh with boundary
        v,f = gpy.read_mesh("test/unit_tests_data/mountain.obj")

        vu,fu = gpy.subdivide(v,f)
        vu_gt,fu_gt = gpy.read_mesh("test/unit_tests_data/upsampled_mountain.obj")
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())

        vu,fu = gpy.subdivide(v,f,method='loop')
        vu_gt,fu_gt = gpy.read_mesh("test/unit_tests_data/looped_mountain.obj")
        self.assertTrue(np.isclose(vu,vu_gt).all())
        self.assertTrue((fu==fu_gt).all())


    def consistency(self,v,f):
        for iters in range(3):
            vu,fu = gpy.subdivide(v,f,method='upsample',iters=iters)
            vus,fus,S = gpy.subdivide(v,f,method='upsample',
                return_matrix=True,iters=iters)
            self.assertTrue(np.isclose(vu,vus).all())
            self.assertTrue((fu==fus).all())
            self.assertTrue(np.isclose(S*v,vus).all())

            if f.shape[1]==3:
                vu,fu = gpy.subdivide(v,f,method='loop',iters=iters)
                vus,fus,S = gpy.subdivide(v,f,method='loop',
                    return_matrix=True,iters=iters)
                self.assertTrue(np.isclose(vu,vus).all())
                self.assertTrue((fu==fus).all())
                self.assertTrue(np.isclose(S*v,vus).all())


if __name__ == '__main__':
    unittest.main()
