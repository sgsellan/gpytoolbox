// Triangulates a polygon with CDT (https://github.com/artem-ogre/CDT), which
// provides the constrained Delaunay triangulation but does no refinement, so
// the maximum area and minimum angle constraints are enforced here, by
// Ruppert's algorithm.
//
// Everything below that implements that refinement is only here because CDT
// does not do it. CDT is the better place for it: it can refine incrementally,
// where we have to retriangulate in batches, which is what forces the extra
// care taken below. So if CDT ever gains refinement, all of it should go, and
// this should become a call into CDT like the triangulation itself already is.
//
// The refinement follows:
//   J. Ruppert, "A Delaunay Refinement Algorithm for Quality 2-Dimensional
//   Mesh Generation", Journal of Algorithms 18(3), 1995, pp. 548-585,
// in the formulation of
//   J. R. Shewchuk, "Delaunay Refinement Algorithms for Triangular Mesh
//   Generation", Computational Geometry 22(1-3), 2002, pp. 21-74.
//
// The polygon is the region bounded by the constraint edges, which CDT keeps
// in the triangulation and which it uses to tell the inside of the polygon
// from its holes. Refinement then repeats two steps until every triangle
// satisfies the constraints:
//
//  - A triangle that is too large or too sharp is fixed by inserting its
//    circumcenter, which is the point that is locally as far as possible from
//    the vertices that made it bad. Because the mesh is Delaunay its
//    circumcircle is empty, so the circumcenter is at least one circumradius
//    away from every existing vertex: inserting it cannot create a shorter
//    edge than the ones already there, which is what keeps the refinement from
//    degenerating and is why the algorithm terminates.
//
//  - A circumcenter that falls outside the polygon would be inserted in the
//    wrong place. Ruppert's observation is that such a point always falls
//    inside the diametral circle of a boundary segment, so instead of
//    inserting a point that "encroaches" upon a segment in this way, the
//    segment itself is split at its midpoint. This is also what makes the
//    triangulation conform to the boundary, and is the only thing that ever
//    adds a vertex on the boundary, so it is skipped when Steiner points on
//    the boundary are not allowed.
//
// Ruppert's algorithm is sequential: it inserts one point and retriangulates.
// CDT is finalized by its erase methods, so retriangulating is done in batches
// here instead. Inserting every circumcenter of a batch at once would break
// the separation the sequential version guarantees, and the refinement would
// then create slivers faster than it removes them, so the worst triangles are
// treated first and a circumcenter is only accepted if it is at least its own
// circumradius away from the others already accepted in the same pass. The
// rest are deferred to the next pass, following the independent-set idea of
//   A. N. Chernikov and N. P. Chrisochoides, "Practical and Efficient Point
//   Insertion Scheduling Method for Parallel Guaranteed Quality Delaunay
//   Refinement", International Conference on Supercomputing, 2004, pp. 48-57.

#include "triangulate_polygon.h"

#include <CDT.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

typedef CDT::V2d<double> Point;

// Refinement stops at whichever of these it hits first, so that a constraint
// that cannot be attained does not loop forever
const int max_refinement_passes = 100;
const std::size_t max_refinement_vertices = 1000000;

// Twice the signed area of the triangle (p0,p1,p2)
double doublearea(const Point& p0, const Point& p1, const Point& p2) {
    return (p1.x-p0.x)*(p2.y-p0.y) - (p1.y-p0.y)*(p2.x-p0.x);
}

// Squared distance between the points u and v
double sqdist(const Point& u, const Point& v) {
    return (u.x-v.x)*(u.x-v.x) + (u.y-v.y)*(u.y-v.y);
}

// Whether the smallest angle of the triangle (p0,p1,p2) is below the angle
// whose squared cosine is cos2q.
bool too_sharp(const Point& p0, const Point& p1, const Point& p2,
    const double cos2q) {
    double l[3] = {sqdist(p1,p2), sqdist(p2,p0), sqdist(p0,p1)};
    std::sort(l,l+3);
    // The smallest angle is the one opposite l[0], and is obtuse if s<=0
    const double s = l[1]+l[2]-l[0];
    return s>0. && s*s > 4.*l[1]*l[2]*cos2q;
}

// Circumcenter of the triangle (p0,p1,p2), in the form given in
// https://en.wikipedia.org/wiki/Circumcircle#Cartesian_coordinates_2
// The caller only ever passes triangles of nonzero area, so d is never zero
Point circumcenter(const Point& p0, const Point& p1, const Point& p2) {
    const double ax = p1.x-p0.x, ay = p1.y-p0.y;
    const double bx = p2.x-p0.x, by = p2.y-p0.y;
    const double na = ax*ax+ay*ay, nb = bx*bx+by*by;
    const double d = 2.*(ax*by-ay*bx);
    return Point(p0.x+(by*na-ay*nb)/d, p0.y+(ax*nb-bx*na)/d);
}

// Distance between the points u and v
double dist(const Point& u, const Point& v) {
    return std::sqrt((u.x-v.x)*(u.x-v.x) + (u.y-v.y)*(u.y-v.y));
}

// Midpoint of the segment (u,v)
Point midpoint(const Point& u, const Point& v) {
    return Point(0.5*(u.x+v.x), 0.5*(u.y+v.y));
}

// Whether p lies strictly inside the diametral circle of the segment (u,v),
// i.e. whether the angle of (u,p,v) at p is obtuse. A dot product is accurate
// to about eps times the product of the two lengths, so anything smaller than
// that counts as zero, and a p on the circle does not encroach
bool encroaches(const Point& u, const Point& v, const Point& p) {
    const double ax = u.x-p.x, ay = u.y-p.y;
    const double bx = v.x-p.x, by = v.y-p.y;
    const double dot = ax*bx + ay*by;
    const double scale = std::sqrt((ax*ax+ay*ay)*(bx*bx+by*by));
    return dot < -8.*std::numeric_limits<double>::epsilon()*scale;
}

}

void triangulate_polygon(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const double a,
    const double q,
    const bool steiner,
    Eigen::MatrixXd& V2,
    Eigen::MatrixXi& F2) {
    std::vector<Point> pts;
    pts.reserve(V.rows());
    for(int i=0; i<V.rows(); ++i) {
        pts.push_back(Point(V(i,0),V(i,1)));
    }
    std::vector<CDT::Edge> segs;
    segs.reserve(F.rows());
    for(int i=0; i<F.rows(); ++i) {
        segs.push_back(CDT::Edge(CDT::VertInd(F(i,0)),CDT::VertInd(F(i,1))));
    }
    const bool refining = a>0. || q>0.;
    const double cos2q = std::cos(q)*std::cos(q);

    // A vertex closer than this to a constraint edge counts as lying on it,
    // and splits it there
    const double tol = V.rows()>0
        ? 1e-12*(V.colwise().maxCoeff()-V.colwise().minCoeff()).norm() : 0.;

    CDT::Triangulation<double> cdt;
    for(int pass=0; pass<max_refinement_passes; ++pass) {
        // The erase methods finalize CDT, so each pass retriangulates from
        // scratch. insertVertices and insertEdges neither reorder nor add
        // vertices, so cdt.vertices is exactly pts
        cdt = CDT::Triangulation<double>(CDT::VertexInsertionOrder::Auto,
            CDT::IntersectingConstraintEdges::NotAllowed, tol);
        // CDT throws on a duplicate vertex and on intersecting constraint
        // edges, both of which are errors in the input
        try {
            cdt.insertVertices(pts);
            if(segs.empty()) {
                cdt.eraseSuperTriangle();
            } else {
                // insertEdges keeps every segment whole. conformToEdges would
                // split them, which is why it is never used: every boundary
                // split is ours, below, and gated on steiner
                cdt.insertEdges(segs);
                cdt.eraseOuterTrianglesAndHoles();
            }
        } catch(const CDT::Error& e) {
            throw std::invalid_argument(std::string("CDT could not triangulate "
                "this polygon. Its vertices must all be distinct, and its edges "
                "must not intersect each other except at their endpoints. CDT "
                "reports: ") + e.what());
        }
        if(!refining || pts.size()>=max_refinement_vertices) {
            break;
        }

        // Without constraint edges the domain is the convex hull, so make its
        // boundary explicit to keep the refinement from growing it
        if(segs.empty()) {
            for(std::size_t t=0; t<cdt.triangles.size(); ++t) {
                for(int k=0; k<3; ++k) {
                    if(cdt.triangles[t].neighbors[k]==CDT::noNeighbor) {
                        segs.push_back(CDT::Edge(
                            cdt.triangles[t].vertices[k],
                            cdt.triangles[t].vertices[(k+1)%3]));
                    }
                }
            }
        }

        std::unordered_map<CDT::Edge,std::size_t> seg_index;
        for(std::size_t s=0; s<segs.size(); ++s) {
            seg_index[segs[s]] = s;
        }
        std::vector<bool> to_split(segs.size(), false);
        std::vector<Point> to_insert;

        // Cocircular triangles share a circumcenter, so the same point can be
        // proposed twice in a pass (CDT rejects duplicate vertices).
        std::set<std::pair<double,double> > known;
        for(std::size_t i=0; i<pts.size(); ++i) {
            known.insert(std::make_pair(pts[i].x,pts[i].y));
        }

        // Ruppert's algorithm: split every segment that a vertex of the mesh
        // encroaches upon. Such a vertex is always the apex of a triangle
        // adjacent to the segment, so this local test suffices.
        for(std::size_t t=0; t<cdt.triangles.size(); ++t) {
            const CDT::VerticesArr3& tv = cdt.triangles[t].vertices;
            for(int k=0; k<3; ++k) {
                typedef std::unordered_map<CDT::Edge,std::size_t> SegIndex;
                const SegIndex::const_iterator s =
                    seg_index.find(CDT::Edge(tv[k],tv[(k+1)%3]));
                if(s!=seg_index.end() && encroaches(
                    cdt.vertices[tv[k]], cdt.vertices[tv[(k+1)%3]],
                    cdt.vertices[tv[(k+2)%3]])) {
                    to_split[s->second] = true;
                }
            }
        }

        // Collect the circumcenter of every triangle that violates the
        // area or the angle constraint, unless that circumcenter encroaches
        // upon segments, which are split instead
        std::vector<Point> candidates;
        std::vector<std::pair<double,std::size_t> > by_radius;
        for(std::size_t t=0; t<cdt.triangles.size(); ++t) {
            const CDT::VerticesArr3& tv = cdt.triangles[t].vertices;
            const Point& p0 = cdt.vertices[tv[0]];
            const Point& p1 = cdt.vertices[tv[1]];
            const Point& p2 = cdt.vertices[tv[2]];
            const double da = doublearea(p0,p1,p2);
            if(da<=0.) {
                continue;
            }
            // A triangle meeting both constraints is left alone: if they all
            // do, nothing is inserted and the loop stops at the end of the pass
            if((a<=0. || 0.5*da<=a) && (q<=0. || !too_sharp(p0,p1,p2,cos2q))) {
                continue;
            }
            const Point c = circumcenter(p0,p1,p2);
            bool encroached = false;
            for(std::size_t s=0; s<segs.size(); ++s) {
                if(encroaches(pts[segs[s].v1()],pts[segs[s].v2()],c)) {
                    encroached = true;
                    to_split[s] = true;
                }
            }
            if(encroached) {
                continue;
            }
            // Sorting by decreasing circumradius treats the worst first
            by_radius.push_back(std::make_pair(-dist(c,p0),candidates.size()));
            candidates.push_back(c);
        }

        // Keep only the circumcenters that are at least their own circumradius
        // apart, so that the batch stays as well separated as it would be if
        // the points were inserted one at a time
        std::sort(by_radius.begin(), by_radius.end());
        for(std::size_t i=0; i<by_radius.size(); ++i) {
            const Point& c = candidates[by_radius[i].second];
            const double r = -by_radius[i].first;
            bool too_close = false;
            for(std::size_t j=0; j<to_insert.size() && !too_close; ++j) {
                too_close = dist(c,to_insert[j])<r;
            }
            if(!too_close && known.insert(std::make_pair(c.x,c.y)).second) {
                to_insert.push_back(c);
            }
        }

        // Apply the pass. Points are only ever appended, so the indices the
        // segments are made of stay valid
        bool changed = false;
        if(steiner) {
            const std::size_t nsegs = segs.size();
            for(std::size_t s=0; s<nsegs; ++s) {
                if(!to_split[s]) {
                    continue;
                }
                const CDT::VertInd v1 = segs[s].v1(), v2 = segs[s].v2();
                const Point mid = midpoint(pts[v1],pts[v2]);
                if(!known.insert(std::make_pair(mid.x,mid.y)).second) {
                    continue;
                }
                const CDT::VertInd vm = CDT::VertInd(pts.size());
                pts.push_back(mid);
                segs[s] = CDT::Edge(v1,vm);
                segs.push_back(CDT::Edge(vm,v2));
                changed = true;
            }
        }
        for(std::size_t i=0; i<to_insert.size(); ++i) {
            pts.push_back(to_insert[i]);
            changed = true;
        }
        // Nothing was refined, so every later pass would refine nothing too
        if(!changed) {
            break;
        }
    }

    // Which vertices the triangles use
    const std::size_t nV = std::size_t(V.rows());
    std::vector<bool> referenced(cdt.vertices.size(), false);
    for(std::size_t t=0; t<cdt.triangles.size(); ++t) {
        for(int k=0; k<3; ++k) {
            referenced[cdt.triangles[t].vertices[k]] = true;
        }
    }

    // A vertex of V that none of them use was only covered by triangles that
    // were erased, so it lies outside the polygon or inside one of its holes
    for(std::size_t i=0; i<nV; ++i) {
        if(!referenced[i]) {
            throw std::invalid_argument("vertex " + std::to_string(i) +
                " of V lies outside the polygon, or inside one of its holes, "
                "so the triangulation cannot contain it. Every vertex of V "
                "must lie in the region that the polygon encloses.");
        }
    }

    // Any other unused vertex is a circumcenter that escaped the polygon, so
    // drop it. Every vertex of V survives, so V's vertices keep their indices
    std::vector<int> remap(cdt.vertices.size(), -1);
    std::size_t n = 0;
    for(std::size_t i=0; i<cdt.vertices.size(); ++i) {
        if(referenced[i]) {
            remap[i] = int(n++);
        }
    }

    V2.resize(n,2);
    for(std::size_t i=0; i<cdt.vertices.size(); ++i) {
        if(remap[i]>=0) {
            V2.row(remap[i]) << cdt.vertices[i].x, cdt.vertices[i].y;
        }
    }
    F2.resize(cdt.triangles.size(),3);
    for(std::size_t i=0; i<cdt.triangles.size(); ++i) {
        F2.row(i) << remap[cdt.triangles[i].vertices[0]],
            remap[cdt.triangles[i].vertices[1]],
            remap[cdt.triangles[i].vertices[2]];
    }
}
