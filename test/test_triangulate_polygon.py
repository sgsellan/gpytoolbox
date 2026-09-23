import hashlib
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
    return {"rectangle":_rectangle(), "circle":_circle(),
            "circle2":_circle(100,0.3), "annulus":_annulus(),
            "annulus2":_annulus(64,1.,0.2), "arch":_arch()}

# The a and q the stored triangulations were made with. Every edge of the arch
# is kept off a whole number of sqrt(a) long: on one, the pre-split inside
# triangulate_polygon cuts it into a different number of pieces on a machine
# that rounds the other way, and the mesh is no longer the stored one
_REGRESSION_PARAMETERS = {"rectangle":(0.05,20.*np.pi/180.),
    "circle":(0.1,25.*np.pi/180.), "circle2":(0.005,20.*np.pi/180.),
    "annulus":(0.02,24.*np.pi/180.), "annulus2":(0.03,15.*np.pi/180.),
    "arch":(0.008,26.*np.pi/180.)}

# How far apart two of these meshes may be and still count as the same
_TOL = 1e-9

# The mesh as the coordinates of its sorted triangles, which is what the
# regression test compares: neither the numbering of the vertices nor the
# order the triangles come in is the same on every platform
def _sorted_mesh(V,F):
    # Weird axis to sort along to avoid symmetry problems
    sort_axis = np.array([1.,0.7548776662466927])
    T = V[F]
    T = T[np.arange(T.shape[0])[:,None],np.argsort(T@sort_axis,axis=1)]
    return T[np.argsort(T.mean(axis=1)@sort_axis)]

# What a failing regression test reports. The digest lets two platforms be
# compared without anyone having to read a mesh
def _digest(T):
    return hashlib.sha256(
        np.round(T/_TOL).astype(np.int64).tobytes()).hexdigest()[:12]

def _triangle(t):
    return " ".join(f"({x:+.9f},{y:+.9f})" for x,y in t)

def _regression_report(name,a,q,computed,stored):
    out = [f"{name}: a={a!r} q={q!r}",
        f"{name}: computed {computed.shape[0]} triangles, "
        f"stored {stored.shape[0]}",
        f"{name}: digests computed/stored "
        f"{_digest(computed)}/{_digest(stored)}"]
    if computed.shape==stored.shape:
        d = np.max(np.abs(computed-stored),axis=(1,2))
        i = int(np.argmax(d))
        out += [f"{name}: {int(np.sum(d>_TOL))} triangles differ, the worst "
            f"of them by {d[i]:.3e}",
            f"{name}: computed {_triangle(computed[i])}",
            f"{name}: stored   {_triangle(stored[i])}"]
    return "\n  ".join(out)


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
            self.assertTrue(np.all(F2>=0))
            self.assertTrue(np.all(F2<V2.shape[0]))
            # No triangle is degenerate, and they are all wound the same way
            self.assertTrue(
                np.all(gpytoolbox.doublearea(V2,F2,signed=True)>0.))
            # No triangle uses the same vertex twice
            self.assertTrue(np.all(F2[:,0]!=F2[:,1]))
            self.assertTrue(np.all(F2[:,1]!=F2[:,2]))
            self.assertTrue(np.all(F2[:,2]!=F2[:,0]))
            self.assertTrue(len(gpytoolbox.non_manifold_edges(F2))==0)
            # The triangles cover the polygon exactly, holes excluded
            self.assertTrue(np.isclose(
                np.sum(0.5*gpytoolbox.doublearea(V2,F2,signed=True)),
                _polygon_area(V,F)))

    def test_regression(self):
        # The triangulation of each polygon has to be the one that was stored
        for name,(V,F) in _polygons().items():
            a,q = _REGRESSION_PARAMETERS[name]
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q)
            with np.load("test/unit_tests_data/"
                f"triangulate_polygon_{name}.npz") as data:
                stored = _sorted_mesh(data["V"],data["F"])
            computed = _sorted_mesh(V2,F2)
            # Sorting the triangles loses the winding, so it is checked here
            self.assertTrue(
                np.all(gpytoolbox.doublearea(V2,F2,signed=True)>0.),name)
            if not (computed.shape==stored.shape
                and np.allclose(computed,stored,rtol=0.,atol=_TOL)):
                self.fail("\n  "
                    +_regression_report(name,a,q,computed,stored))

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
        for name,(V,F) in _polygons().items():
            for a,q in [(0.2,np.pi/12), (0.1,np.pi/8), (0.05,np.pi/12),
                        (0.05,np.pi/8), (0.02,np.pi/8),
                        (0.02,25.*np.pi/180.), (0.005,np.pi/8)]:
                V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q)
                areas = 0.5*gpytoolbox.doublearea(V2,F2,signed=True)
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
            longest = np.max(np.linalg.norm(V[F[:,1],:]-V[F[:,0],:],axis=1))
            a,q = 0.9*longest**2, np.pi/8
            V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q,steiner=False)
            V3,F3 = gpytoolbox.triangulate_polygon(V,F,a=a,q=q,steiner=True)
            polygon_edges = np.sort(F,axis=1)
            kept_without, kept_with = _edge_set(F2), _edge_set(F3)
            self.assertFalse(all(tuple(e) in kept_with
                for e in polygon_edges), name)
            self.assertTrue(all(tuple(e) in kept_without
                for e in polygon_edges), name)
            self.assertTrue(np.allclose(V2[:V.shape[0],:],V), name)
            self.assertTrue(np.allclose(V3[:V.shape[0],:],V), name)
            self.assertTrue(V3.shape[0]>=V2.shape[0], name)

        V,F = _rectangle()
        V2,F2 = gpytoolbox.triangulate_polygon(V,F,a=0.02,q=np.pi/8,
            steiner=False)
        V3,F3 = gpytoolbox.triangulate_polygon(V,F,a=0.02,q=np.pi/8,
            steiner=True)
        self.assertTrue(np.max(0.5*gpytoolbox.doublearea(V2,F2))>0.02)
        self.assertTrue(
            np.max(0.5*gpytoolbox.doublearea(V3,F3))<=0.02*(1.+1e-10))
        self.assertTrue(V3.shape[0]>V2.shape[0])

    def test_no_edges_is_convex_hull(self):
        # Without edges the convex hull of the points is triangulated, so for a
        # convex polygon that is the polygon itself
        V,F = _circle()
        V2,F2 = gpytoolbox.triangulate_polygon(V,a=0.,q=0.)
        self.assertTrue(V2.shape[0]==V.shape[0])
        self.assertTrue(np.isclose(
            np.sum(0.5*gpytoolbox.doublearea(V2,F2,signed=True)),
            _polygon_area(V,F)))

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
