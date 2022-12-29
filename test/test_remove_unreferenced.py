from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest


class TestRemoveUnreferenced(unittest.TestCase):

    def test_polyline(self):
        V = np.array([[0],[0.2],[0.5],[0.98],[1.0],[1.4],[2.0],[0.3],
            [0.7],[1.2],[4.5]])
        F = np.array([[0,1],[1,2],[2,5],[5,3],[3,9]])
        newV,newF,I,J = gpy.remove_unreferenced(V, F, return_maps=True)
        newV1,newF1 = gpy.remove_unreferenced(V, F)
        newV2,newF2 = gpy.remove_unreferenced(None, F)
        self.assertTrue((newV == newV1).all())
        self.assertTrue(newV2 is None)
        self.assertTrue((newF == newF1).all())
        self.assertTrue((newF == newF2).all())
        self.assertTrue(np.all(newV == np.array([[0.  ],
            [0.2 ],
            [0.5 ],
            [0.98],
            [1.4 ],
            [1.2 ]])))
        self.assertTrue(np.all(newF == np.array([[0, 1],
            [1, 2],
            [2, 4],
            [4, 3],
            [3, 5]])))
        self.assertTrue(np.all(I == np.array([ 0,
            1,
            2,
            3,
            -1,
            4,
            -1,
            -1,
            -1,
            5,
            -1])))
        self.assertTrue(np.all(J == np.array([0,
            1,
            2,
            3,
            5,
            9])))

        k = 5
        n = 200
        m = 20
        rng = np.random.default_rng()
        for i in range(k):
            V = rng.random((n,2))
            F = rng.integers(-1, n, size=(m,2))
            newV,newF,I,J = gpy.remove_unreferenced(V, F, return_maps=True)
            newV1,newF1 = gpy.remove_unreferenced(V, F)
            newV2,newF2 = gpy.remove_unreferenced(None, F)
            self.assertTrue((newV == newV1).all())
            self.assertTrue(newV2 is None)
            self.assertTrue((newF == newF1).all())
            self.assertTrue((newF == newF2).all())
            self.consistency(V, F, newV, newF, I, J)

    def test_triangle_mesh(self):
        V,F = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        newV,newF,I,J = gpy.remove_unreferenced(V, F, return_maps=True)
        newV1,newF1 = gpy.remove_unreferenced(V, F)
        newV2,newF2 = gpy.remove_unreferenced(None, F)
        self.assertTrue((newV == newV1).all())
        self.assertTrue(newV2 is None)
        self.assertTrue((newF == newF1).all())
        self.assertTrue((newF == newF2).all())
        self.assertTrue(np.all(V == newV))
        self.assertTrue(np.all(F == newF))

        k = 5
        n = 200
        m = 20
        rng = np.random.default_rng()
        for i in range(k):
            V = rng.random((n,3))
            F = rng.integers(-1, n, size=(m,3))
            newV,newF,I,J = gpy.remove_unreferenced(V, F, return_maps=True)
            newV1,newF1 = gpy.remove_unreferenced(V, F)
            newV2,newF2 = gpy.remove_unreferenced(None, F)
            self.assertTrue((newV == newV1).all())
            self.assertTrue(newV2 is None)
            self.assertTrue((newF == newF1).all())
            self.assertTrue((newF == newF2).all())
            self.consistency(V, F, newV, newF, I, J)

    def test_tet_mesh(self):
        k = 5
        n = 200
        m = 20
        rng = np.random.default_rng()
        for i in range(k):
            V = rng.random((n,3))
            F = rng.integers(-1, n, size=(m,4))
            newV,newF,I,J = gpy.remove_unreferenced(V, F, return_maps=True)
            newV1,newF1 = gpy.remove_unreferenced(V, F)
            newV2,newF2 = gpy.remove_unreferenced(None, F)
            self.assertTrue((newV == newV1).all())
            self.assertTrue(newV2 is None)
            self.assertTrue((newF == newF1).all())
            self.assertTrue((newF == newF2).all())
            self.consistency(V, F, newV, newF, I, J)


    def consistency(self,V,F,newV,newF,I,J):
        if V is not None:
            self.assertTrue(np.all(V[J,:] == newV))
        self.assertTrue(np.all(I[F] == newF))


if __name__ == '__main__':
    unittest.main()
