import numpy as np
from .context import gpytoolbox
from .context import unittest


# Vertices are snapped to this grid, so every coordinate is exact in binary
# floating point.
_GRID = 2.**20

def _snap(V):
    return np.round(V*_GRID)/_GRID


# Every test polygon is one or more closed loops: outer loops run
# counterclockwise, the loops bounding a hole clockwise.
def _rectangle():
    V = np.array([[0.,0.],[2.,0.],[2.,1.],[0.,1.]])
    F = np.array([[0,1],[1,2],[2,3],[3,0]])
    return V,F

def _circle(n=32,r=1.):
    th = np.linspace(0.,2.*np.pi,n,endpoint=False)
    V = np.stack((r*np.cos(th),r*np.sin(th)),axis=-1)
    F = np.stack((np.arange(n),(np.arange(n)+1)%n),axis=-1)
    return _snap(V),F

def _annulus(n=32,ro=1.,ri=0.5):
    Vo,Fo = _circle(n,ro)
    Vi,Fi = _circle(n,ri)
    # The inner loop is reversed so that it bounds a hole
    return np.vstack((Vo,Vi)), np.vstack((Fo,np.fliplr(Fi)+n))

def _arch(n=16,ro=1.,ri=0.6):
    # A semicircular band: the outer arc, the inner arc back, and two feet
    tho = np.linspace(0.,np.pi,n)
    thi = np.linspace(np.pi,0.,n)
    V = np.vstack((np.stack((ro*np.cos(tho),ro*np.sin(tho)),axis=-1),
                   np.stack((ri*np.cos(thi),ri*np.sin(thi)),axis=-1)))
    m = V.shape[0]
    F = np.stack((np.arange(m),(np.arange(m)+1)%m),axis=-1)
    return _snap(V),F

def _polygons():
    return {"rectangle":_rectangle(), "circle":_circle(), "circle2":_circle(100,0.3),
            "annulus":_annulus(), "annulus2":_annulus(64,1.,0.2), "arch":_arch()}

# The a and q the stored triangulations were made with.
def _regression_parameters():
    return {"rectangle":(0.05,20.*np.pi/180.),
            "circle":(0.1,25.*np.pi/180.),
            "circle2":(0.005,20.*np.pi/180.),
            "annulus":(0.02,24.*np.pi/180.),
            "annulus2":(0.03,15.*np.pi/180.),
            "arch":(0.008,26.*np.pi/180.)}

def _correct_output_path(name):
    return f"test/unit_tests_data/triangulate_polygon_{name}.npz"


# Signed area of the region bounded by the edges F of the polygon V
def _polygon_area(V,F):
    return 0.5*np.sum(V[F[:,0],0]*V[F[:,1],1] - V[F[:,1],0]*V[F[:,0],1])

# Set of the unoriented edges of the mesh with faces F
def _edge_set(F):
    return set(map(tuple,np.unique(np.sort(np.reshape(
        F[:,[0,1,1,2,2,0]],(-1,2)),axis=1),axis=0)))


class TestTriangulatePolygon(unittest.TestCase):
    def test_manifold(self):
        for name,(V,F) in _polygons().items():
            V2,F2 = gpytoolbox.triangulate_polygon(V,F)
            self.assertTrue(V2.ndim==2 and V2.shape[1]==2)
            self.assertTrue(F2.ndim==2 and F2.shape[1]==3)
            self.assertTrue(V2.shape[0]>0)
            self.assertTrue(F2.shape[0]>0)
            # Every index points at a vertex that exists
            self.assertTrue(np.all(F2>=0))
            self.assertTrue(np.all(F2<V2.shape[0]))
            # No triangle is degenerate, and they are all oriented the same way
            self.assertTrue(np.all(0.5*gpytoolbox.doublearea(V2,F2)>0.))
            # No triangle uses the same vertex twice
            self.assertTrue(np.all(F2[:,0]!=F2[:,1]))
            self.assertTrue(np.all(F2[:,1]!=F2[:,2]))
            self.assertTrue(np.all(F2[:,2]!=F2[:,0]))
            self.assertTrue(len(gpytoolbox.non_manifold_edges(F2))==0)
            # The triangles cover the polygon exactly, holes excluded
            self.assertTrue(np.isclose(np.sum(0.5*gpytoolbox.doublearea(V2,F2)),
                _polygon_area(V,F)))

    def test_regression(self):
        # The triangulation of each polygon has to be the one that was stored
        for name,(V,F) in _polygons().items():
            a,q = _regression_parameters()[name]
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q)
            with np.load(_correct_output_path(name)) as data:
                report = (f"\n  {name}: a={a!r} q={q!r}"
                    f"\n  stored    {data['V'].shape[0]} vertices, "
                    f"{data['F'].shape[0]} faces"
                    f"\n  computed  {V2.shape[0]} vertices, {F2.shape[0]} faces")
                self.assertEqual(F2.shape,data["F"].shape,report)
                self.assertTrue(np.all(F2==data["F"]),report)
                self.assertEqual(V2.shape,data["V"].shape,report)
                self.assertTrue(np.allclose(V2,data["V"],rtol=0.,atol=1e-9),
                    report)

    def test_area_argument(self):
        for name,(V,F) in _polygons().items():
            # Each a is half the largest triangle of the mesh before it, so
            # every step has something left to refine
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.,q=0.)
            for step in range(3):
                a = 0.5*np.max(0.5*gpytoolbox.doublearea(V2,F2))
                coarser = V2.shape[0]
                V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=0.)
                # No triangle is larger than a.
                self.assertTrue(
                    np.max(0.5*gpytoolbox.doublearea(V2,F2))<=a*(1.+1e-10))
                self.assertTrue(V2.shape[0]>coarser)
            # a=0. means no constraint at all, so nothing is refined
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.,q=0.)
            self.assertTrue(V2.shape[0]==V.shape[0])

    def test_angle_argument(self):
        for name,(V,F) in _polygons().items():
            Vb,Fb = gpytoolbox.triangulate_polygon(V,F,a=0.,q=0.)
            base_angle = np.min(gpytoolbox.tip_angles(Vb,Fb))
            coarser = Vb.shape[0]
            for q in [np.pi/12,np.pi/8,25.*np.pi/180.]:
                V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.,q=q)
                # No triangle has an angle smaller than q
                self.assertTrue(
                    np.min(gpytoolbox.tip_angles(V2,F2))>=q*(1.-1e-10))
                # A q the unrefined mesh does not meet forces refinement,
                # and a larger q never gives a coarser mesh
                if base_angle<q:
                    self.assertTrue(V2.shape[0]>Vb.shape[0])
                self.assertTrue(V2.shape[0]>=coarser)
                coarser = V2.shape[0]
            # q=0. means no constraint at all, so nothing is refined
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.,q=0.)
            self.assertTrue(V2.shape[0]==V.shape[0])

    def test_area_and_angle_arguments_together(self):
        # Both limits have to hold at the same time. Only the limits
        # themselves are asserted: the mesh is not monotone in a and q, since
        # the area pass changes the order the angle pass inserts in
        for name,(V,F) in _polygons().items():
            for a,q in [(0.2,np.pi/12), (0.1,np.pi/8), (0.05,np.pi/12),
                        (0.05,np.pi/8), (0.02,np.pi/8),
                        (0.02,25.*np.pi/180.), (0.005,np.pi/8)]:
                V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q)
                areas = 0.5*gpytoolbox.doublearea(V2,F2)
                self.assertTrue(np.max(areas)<=a*(1.+1e-10))
                self.assertTrue(
                    np.min(gpytoolbox.tip_angles(V2,F2))>=q*(1.-1e-10))
                # It is still a valid mesh of the polygon
                self.assertTrue(np.all(areas>0.))
                self.assertTrue(len(gpytoolbox.non_manifold_edges(F2))==0)
                self.assertTrue(np.isclose(np.sum(areas),_polygon_area(V,F)))
                self.assertTrue(np.array_equal(np.unique(F2),
                    np.arange(V2.shape[0])))
                self.assertTrue(np.allclose(V2[:V.shape[0],:],V))

    def test_area_and_angle_arguments_are_both_active(self):
        # Each limit on its own leaves the other one violated, so asking for
        # both is doing the work of both
        V,F = _circle(64,1.)
        a,q = 0.02,np.pi/8
        Va,Fa = gpytoolbox.triangulate_polygon(V,F,a=a,q=0.)
        self.assertTrue(np.max(0.5*gpytoolbox.doublearea(Va,Fa))<=a*(1.+1e-10))
        self.assertTrue(np.min(gpytoolbox.tip_angles(Va,Fa))<q)
        Vq,Fq = gpytoolbox.triangulate_polygon(V,F,a=0.,q=q)
        self.assertTrue(np.min(gpytoolbox.tip_angles(Vq,Fq))>=q*(1.-1e-10))
        self.assertTrue(np.max(0.5*gpytoolbox.doublearea(Vq,Fq))>a)
        V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q)
        self.assertTrue(np.max(0.5*gpytoolbox.doublearea(V2,F2))<=a*(1.+1e-10))
        self.assertTrue(np.min(gpytoolbox.tip_angles(V2,F2))>=q*(1.-1e-10))

    def test_steiner_argument(self):
        for name,(V,F) in _polygons().items():
            # a has to be small enough that the boundary is too coarse for
            # it, or both calls would return the same mesh
            longest = np.max(np.linalg.norm(V[F[:,1],:]-V[F[:,0],:],axis=1))
            a,q = 0.9*longest**2, np.pi/8
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q,steiner=False)
            V3,F3 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q,steiner=True)
            polygon_edges = np.sort(F,axis=1)
            kept_without, kept_with = _edge_set(F2), _edge_set(F3)
            # Allowing them, the boundary is split: without this the rest of
            # the test would hold whether steiner did anything or not
            self.assertFalse(all(tuple(e) in kept_with
                for e in polygon_edges), name)
            # Forbidding them, every edge of the polygon survives
            self.assertTrue(all(tuple(e) in kept_without
                for e in polygon_edges), name)
            # Either way the polygon's vertices are the first ones, in the
            # order they were given, and allowing splits only adds vertices
            self.assertTrue(np.allclose(V2[:V.shape[0],:],V), name)
            self.assertTrue(np.allclose(V3[:V.shape[0],:],V), name)
            self.assertTrue(V3.shape[0]>=V2.shape[0], name)

        # What it costs: the rectangle's four edges are long, and a triangle
        # against one of them cannot meet a until it is split, so without
        # Steiner points the area constraint is out of reach
        V,F = _rectangle()
        V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.02,q=np.pi/8,
            steiner=False)
        V3,F3 = gpytoolbox.triangulate_polygon(V,F,a=0.02,q=np.pi/8,
            steiner=True)
        self.assertTrue(np.max(0.5*gpytoolbox.doublearea(V2,F2))>0.02)
        self.assertTrue(np.max(0.5*gpytoolbox.doublearea(V3,F3))<=0.02*(1.+1e-10))
        self.assertTrue(V3.shape[0]>V2.shape[0])

    def test_no_edges_is_convex_hull(self):
        # Without edges the convex hull of the points is triangulated, so for a
        # convex polygon that is the polygon itself
        V,F = _circle()
        V2,F2 = gpytoolbox.triangulate_polygon(V,a=0.,q=0.)
        self.assertTrue(V2.shape[0]==V.shape[0])
        self.assertTrue(np.isclose(np.sum(0.5*gpytoolbox.doublearea(V2,F2)),_polygon_area(V,F)))

    def test_every_output_vertex_is_used(self):
        # No vertex of the output is left without a triangle using it
        for name,(V,F) in _polygons().items():
            for kw in [dict(a=0.,q=0.), dict(a=0.05,q=np.pi/8),
                       dict(a=0.02,q=np.pi/8,steiner=False)]:
                V2,F2 = gpytoolbox.triangulate_polygon(V,F,**kw)
                self.assertTrue(np.array_equal(np.unique(F2),
                    np.arange(V2.shape[0])))

    def test_points_are_kept(self):
        # Points of V that no edge of F refers to are triangulated along with
        # the polygon
        V,F = _rectangle()
        loose = np.array([[0.5,0.5],[1.,0.25],[1.5,0.75]])
        Vp = np.vstack((V,loose))
        self.assertTrue(np.all(F<V.shape[0]))
        for a,q in [(0.,0.),(0.05,np.pi/8)]:
            V2,F2 = gpytoolbox.triangulate_polygon(Vp,F,a=a,q=q)
            # The input vertices are all there, unmoved and in the same order
            self.assertTrue(np.array_equal(V2[:Vp.shape[0],:],Vp))
            # Every one of them is used by a triangle, loose or not
            for i in range(Vp.shape[0]):
                self.assertTrue(np.any(F2==i))
            self.assertTrue(np.array_equal(np.unique(F2),
                np.arange(V2.shape[0])))

    def test_points_with_no_triangle_are_an_error(self):
        # A vertex no triangle can use is an error, not an orphan in the output
        V,F = _rectangle()
        with self.assertRaises(ValueError) as e:
            gpytoolbox.triangulate_polygon(np.vstack((V,[[3.,3.]])),F)
        self.assertTrue("outside the polygon" in str(e.exception))
        # A point inside a hole is such a vertex
        V,F = _annulus()
        for p in [[0.,0.],[0.25,0.],[0.,-0.3]]:
            with self.assertRaises(ValueError) as e:
                gpytoolbox.triangulate_polygon(np.vstack((V,[p])),F)
            self.assertTrue("inside one of its holes" in str(e.exception))

    def test_points_on_a_hole_boundary_are_not_an_error(self):
        # A point on the boundary of a hole is on the polygon, not in the
        # hole, and a rounding error to either side must not change that
        V,F = _annulus()
        inner = V[V.shape[0]//2:,:]
        mid = 0.5*(inner[0,:]+inner[1,:])
        for eps in [0.,1e-16,-1e-16,1e-15,-1e-15,1e-13,-1e-13]:
            Vp = np.vstack((V,mid*(1.+eps)))
            V2,F2 = gpytoolbox.triangulate_polygon(Vp,F,a=0.,q=0.)
            # The point is kept where it was put, and used by a triangle
            self.assertTrue(np.array_equal(V2[:Vp.shape[0],:],Vp))
            self.assertTrue(np.any(F2==V.shape[0]))
            self.assertTrue(np.array_equal(np.unique(F2),
                np.arange(V2.shape[0])))

    def test_cdt_errors(self):
        V,F = _rectangle()
        # A polygon that repeats a vertex
        with self.assertRaises(ValueError) as e:
            gpytoolbox.triangulate_polygon(np.vstack((V,V[0,:])),F)
        self.assertTrue("CDT" in str(e.exception))
        self.assertTrue("duplicate" in str(e.exception).lower())
        # A polygon whose edges cross each other away from their endpoints
        with self.assertRaises(ValueError) as e:
            gpytoolbox.triangulate_polygon(V,np.array([[0,2],[1,3]]))
        self.assertTrue("CDT" in str(e.exception))
        # Whatever went wrong, the function is still usable afterwards
        V2,F2 = gpytoolbox.triangulate_polygon(V,F)
        self.assertTrue(F2.shape[0]>0)

    def test_invalid_input(self):
        V,F = _rectangle()
        # Arguments that are not two-dimensional, or out of range
        with self.assertRaises(ValueError):
            gpytoolbox.triangulate_polygon(np.zeros((4,3)))
        with self.assertRaises(ValueError):
            gpytoolbox.triangulate_polygon(V,np.zeros((2,3),dtype=int))
        with self.assertRaises(ValueError):
            gpytoolbox.triangulate_polygon(V,F,a=-1.)
        with self.assertRaises(ValueError):
            gpytoolbox.triangulate_polygon(V,F,q=np.pi)


if __name__ == '__main__':
    unittest.main()
