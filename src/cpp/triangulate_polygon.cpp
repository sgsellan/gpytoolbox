// Triangulates the region a polygon encloses, with CDT
// (https://github.com/artem-ogre/CDT).
//
//  V        the polygon's vertices, one xy per row. A vertex no edge of F
//           refers to is triangulated along with the polygon, which is how a
//           caller puts a point of their own in the output; it has to lie
//           inside the polygon or this throws
//  F        the polygon's edges, two indices into V per row, as many closed
//           loops as needed. A loop inside another one is a hole, a loop
//           inside a hole is filled again. No edges at all means the polygon
//           is the convex hull of V
//  a        the largest area an output triangle may have, 0 for no limit
//  q        the smallest angle an output triangle may have, in radians,
//           0 for no limit
//  steiner  whether a and q may put new vertices on the polygon's boundary.
//           If false, every edge of F is also an edge of the output, and a
//           and q only hold where they can be met without splitting one
//  V2, F2   the output triangle mesh: V2 is V, unmoved and in the same order,
//           followed by whatever the refinement added, and F2 indexes into it
//
// CDT does the constrained Delaunay triangulation and the Delaunay refinement
// that a and q need. What it does not do, and this file adds:
//
//  - CDT refines for one criterion at a time, so a and q alternate here.
//  - CDT always splits a constraint edge encroached upon by a point it
//    inserts, so steiner=false drops the points that caused a split
//    afterwards.
//  - CDT splits by halving, and halves again while the point is still not
//    separated, leaving one piece far shorter than its neighbours and a
//    triangle against it too thin for q. presplit hands it a boundary fine
//    enough that it has nothing left to halve.
//  - Without constraint edges the polygon is the convex hull of V, which has
//    to be spelled out or the refinement cannot reach it.

#include "triangulate_polygon.h"

#include <CDT.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

typedef CDT::V2d<double> Point;
typedef CDT::Triangulation<double> Triangulation;

// So an unattainable constraint does not loop forever
const int max_refinement_passes = 100;
const CDT::VertInd max_refinement_vertices = 1000000;

// The triangles outside the polygon, which the refinement has to leave alone
CDT::TriIndUSet outside(const Triangulation& cdt) {
    return cdt.fixedEdges.empty() ? cdt.collectSuperTriangle()
                                  : cdt.collectOuterTrianglesAndHoles();
}

// Whether p is inside the diametral circle of one of the polygon's edges,
// which is what makes CDT split that edge
bool encroaches(const std::vector<Point>& pts,
    const std::vector<CDT::Edge>& segs, const Point& p) {
    for(std::size_t s=0; s<segs.size(); ++s) {
        if(CDT::detail::isEncroachingOnEdge(p,
            pts[segs[s].v1()], pts[segs[s].v2()])) {
            return true;
        }
    }
    return false;
}

// segs cut into pieces no longer than the side of a triangle of area a, with
// the points that cut them appended to pts. Without a there is no length to
// aim for and nothing to do
std::vector<CDT::Edge> presplit(std::vector<Point>& pts,
    const std::vector<CDT::Edge>& segs, const double a) {
    if(a<=0.) {
        return segs;
    }
    // Proxy for the width of a triangle of area a, so this is the boundary resolution
    // that a asks for
    const double target = std::sqrt(a);
    std::vector<CDT::Edge> out;
    for(std::size_t i=0; i<segs.size(); ++i) {
        const Point u = pts[segs[i].v1()], v = pts[segs[i].v2()]; // pts grows
        // n equal pieces, as few as will all come out short enough, and never
        // more points than are left in the budget
        std::size_t n = 1;
        if(pts.size()<max_refinement_vertices) {
            const double pieces = std::ceil(CDT::distance(u,v)/target);
            n = pieces>1. ? std::size_t(std::min(pieces,
                double(max_refinement_vertices-pts.size()))) : 1;
        }
        CDT::VertInd prev = segs[i].v1();
        for(std::size_t k=1; k<n; ++k) {
            const double t = double(k)/double(n);
            pts.push_back(Point(u.x+t*(v.x-u.x), u.y+t*(v.y-u.y)));
            out.push_back(CDT::Edge(prev,CDT::VertInd(pts.size()-1)));
            prev = CDT::VertInd(pts.size()-1);
        }
        out.push_back(CDT::Edge(prev,segs[i].v2()));
    }
    return out;
}

// CDT throws on a duplicate vertex and on intersecting constraint edges, both
// of which are errors in the input
Triangulation build(const std::vector<Point>& pts,
    const std::vector<CDT::Edge>& segs, const double tol) {
    Triangulation cdt(CDT::VertexInsertionOrder::Auto,
        CDT::IntersectingConstraintEdges::NotAllowed, tol);
    try {
        cdt.insertVertices(pts);
        if(!segs.empty()) {
            // insertEdges keeps the edges whole, conformToEdges would split
            cdt.insertEdges(segs);
        }
    } catch(const CDT::Error& e) {
        throw std::invalid_argument(std::string("CDT could not triangulate "
            "this polygon. Its vertices must all be distinct, and its edges "
            "must not intersect each other except at their endpoints. CDT "
            "reports: ") + e.what());
    }
    return cdt;
}

// pts and segs refined to meet a and q, erased to the region the polygon
// encloses. Its vertices are pts, followed by the inserted ones
Triangulation triangulate(const std::vector<Point>& pts,
    const std::vector<CDT::Edge>& segs, const double a, const double q,
    const double tol, const double min_edge_length) {
    Triangulation cdt = build(pts,segs,tol);
    // Collected, not erased: erasing finalizes, and refining needs it
    // unfinalized. CDT keeps the set up to date
    CDT::TriIndUSet erased = outside(cdt);
    for(int pass=0; pass<max_refinement_passes && (a>0. || q>0.); ++pass) {
        const std::size_t before = cdt.vertices.size();
        const std::size_t used = before-CDT::nSuperTriVerts;
        if(used>=max_refinement_vertices) {
            break;
        }
        const CDT::VertInd budget = CDT::VertInd(max_refinement_vertices-used);
        // One criterion per call, a as the largest area allowed and q as the
        // smallest angle. Each call resumes where the last left off, so a pass
        // that inserts nothing had nothing left for either, and their order
        // does not matter
        if(a>0.) {
            cdt.refineTriangles(budget, CDT::RefinementCriterion::LargestArea,
                a, &erased, min_edge_length);
        }
        if(q>0.) {
            cdt.refineTriangles(budget, CDT::RefinementCriterion::SmallestAngle,
                q, &erased, min_edge_length);
        }
        if(cdt.vertices.size()==before) {
            break;
        }
    }
    cdt.finalizeTriangulation(erased);
    return cdt;
}

// The edges of the convex hull of pts, as indices into pts
std::vector<CDT::Edge> hull_edges(const std::vector<Point>& pts,
    const double tol) {
    Triangulation cdt = build(pts,std::vector<CDT::Edge>(),tol);
    cdt.eraseSuperTriangle();
    std::vector<CDT::Edge> hull;
    for(std::size_t i=0; i<cdt.triangles.size(); ++i) {
        const CDT::Triangle& t = cdt.triangles[i];
        for(int k=0; k<3; ++k) {
            if(t.neighbors[k]==CDT::noNeighbor) {
                hull.push_back(CDT::Edge(t.vertices[k],t.vertices[(k+1)%3]));
            }
        }
    }
    return hull;
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
    // V and F in CDT's own types, which is what every step below works on
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
    // With both limits off, the constrained Delaunay triangulation of V and F
    // is already the answer and nothing here has to refine anything
    const bool refining = a>0. || q>0.;

    // The extent of the input is the only scale there is. A vertex within tol
    // of a constraint edge lies on it; an edge or triangle shorter than
    // min_edge_length is left alone, which stops the refinement where a sharp
    // corner makes q unattainable
    const double scale = V.rows()>0
        ? (V.colwise().maxCoeff()-V.colwise().minCoeff()).norm() : 0.;
    const double tol = 1e-12*scale;
    const double min_edge_length = 1e-9*scale;

    // The refinement needs a boundary to conform to, so an F that was empty
    // becomes the hull that the triangulation covers anyway
    if(segs.empty() && refining) {
        segs = hull_edges(pts,tol);
    }
    // pts and segs stay as given: the steiner test below measures against them
    std::vector<Point> refined_pts = pts;
    const std::vector<CDT::Edge> refined_segs =
        refining ? presplit(refined_pts,segs,a) : segs;
    Triangulation cdt =
        triangulate(refined_pts,refined_segs,a,q,tol,min_edge_length);

    if(refining && !steiner) {
        // Keep what the refinement would have added without splitting an
        // edge. presplit's points lie on the edges, so they go too and the
        // boundary is back to the edges given
        std::vector<Point> kept(pts);
        for(std::size_t i=pts.size(); i<cdt.vertices.size(); ++i) {
            if(!encroaches(pts,segs,cdt.vertices[i])) {
                kept.push_back(cdt.vertices[i]);
            }
        }
        cdt = triangulate(kept,segs,0.,0.,tol,min_edge_length);
    }

    // Which vertices the erase step left with a triangle on them
    std::vector<bool> referenced(cdt.vertices.size(), false);
    for(std::size_t t=0; t<cdt.triangles.size(); ++t) {
        for(int k=0; k<3; ++k) {
            referenced[cdt.triangles[t].vertices[k]] = true;
        }
    }
    // A vertex of V no triangle uses was only covered by erased ones
    for(std::size_t i=0; i<std::size_t(V.rows()); ++i) {
        if(!referenced[i]) {
            throw std::invalid_argument("vertex " + std::to_string(i) +
                " of V lies outside the polygon, or inside one of its holes, "
                "so the triangulation cannot contain it. Every vertex of V "
                "must lie in the region that the polygon encloses.");
        }
    }

    // Any other unused vertex is outside the polygon, so drop it. V's
    // vertices all survive, with their indices
    std::vector<int> remap(cdt.vertices.size(), -1);
    int n = 0;
    for(std::size_t i=0; i<cdt.vertices.size(); ++i) {
        if(referenced[i]) {
            remap[i] = n++;
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
