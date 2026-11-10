// Method::PSR: the isosurface extraction of Poisson Surface Reconstruction
// (Kazhdan, Klein, Dalal & Hoppe, "Unconstrained Isosurface Extraction on
// Arbitrary Octrees", 2007), on the octree as built: leaves of any sizes can
// be neighbors.
// - Finest edges: every leaf edge is split at the corners of all the leaves
//   touching it; vertices lie on these, so every leaf touching an edge uses
//   the same ones.
// - Finest faces: a leaf's contour on a face is the one of the finest faces
//   covering it (itself, or the faces of the smaller leaves across), so the
//   two sides of a face always agree.
// - A leaf's contour segments close into polygons, triangulated with minimal
//   area (as in PoissonRecon), never with a diagonal between two vertices on
//   one face of the leaf: the leaf across that face may have both vertices
//   too, and the same diagonal, which would make the edge non-manifold. A
//   polygon that cannot be triangulated so is fanned around its centroid.
#include <limits>

#include "octree_mc_common.h"

namespace omc {
namespace detail {
namespace {

using Segment = std::array<int, 2>;  // directed, between mesh vertices

class PSR {
 public:
  PSR(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt)
      : subdivide_(subdivide), f_(f), tree_(opt) {}

  Mesh run() {
    tree_.build(subdivide_);
    tree_.collect_leaves();
    f_.evaluate_corners(tree_);
    register_edges();
    // The polygons (vertices only identified by their edges), then the
    // vertices placed all at once, then the polygons triangulated
    std::vector<Segment> segments;
    std::vector<std::vector<int>> polygons;
    std::vector<int> polygon_leaf;
    for (int i : tree_.leaves) {
      segments.clear();
      for (int face = 0; face < 6; ++face) leaf_face_segments(i, face, segments);
      loops(segments, polygons);
      polygon_leaf.resize(polygons.size(), i);
    }
    mesh_.vertices = vertices_.place(tree_, f_);
    for (size_t k = 0; k < polygons.size(); ++k) triangulate(polygons[k], polygon_leaf[k]);
    return std::move(mesh_);
  }

 private:
  const SubdivideFn& subdivide_;
  Field f_;
  Octree tree_;
  EdgeVertices vertices_;
  // Every axis-aligned lattice line holding leaf edges -> the leaf corners on
  // it, sorted: they split the leaf edges into finest edges
  std::unordered_map<uint64_t, std::vector<int64_t>> lines_;
  Mesh mesh_;

  // The line through p along axis a
  static uint64_t line_key(const Lattice& p, int a) {
    const int u = (a + 1) % 3, v = (a + 2) % 3;
    return uint64_t(a) | uint64_t(p[u]) << 2 | uint64_t(p[v]) << 23;
  }

  void register_edges() {
    for (int i : tree_.leaves) {
      const Node& n = tree_.nodes[i];
      const int64_t s = tree_.size(n);
      for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 4; ++k) {  // the 4 edges along axis a
          Lattice p = n.x;
          p[(a + 1) % 3] += (k & 1) * s;
          p[(a + 2) % 3] += (k >> 1) * s;
          auto& line = lines_[line_key(p, a)];
          line.push_back(p[a]);
          line.push_back(p[a] + s);
        }
    }
    for (auto& [key, line] : lines_) {
      std::sort(line.begin(), line.end());
      line.erase(std::unique(line.begin(), line.end()), line.end());
    }
  }

  // Appends the lattice points from p to q (an edge along axis a), the leaf
  // corners between them included, q excluded
  void edge_points(const Lattice& p, const Lattice& q, int a, std::vector<Lattice>& out) const {
    const auto& line = lines_.at(line_key(p, a));
    out.push_back(p);
    Lattice x = p;
    if (q[a] > p[a]) {
      for (auto it = std::upper_bound(line.begin(), line.end(), p[a]); it != line.end() && *it < q[a]; ++it) {
        x[a] = *it;
        out.push_back(x);
      }
    } else {
      auto it = std::lower_bound(line.begin(), line.end(), p[a]);
      while (it != line.begin() && *(it - 1) > q[a]) {
        x[a] = *--it;
        out.push_back(x);
      }
    }
  }

  // The iso-segments on a finest face: the square [x, x + len]^2 normal to
  // axis a, its boundary split at the leaf corners on it. Marching squares
  // on that polygon: walking its boundary counter-clockwise about +a, each
  // inside -> outside crossing is joined to the outside -> inside crossing
  // before it. That separates the inside corners, and depends on the face
  // alone, so both leaves sharing the face agree. Segments are oriented for
  // the leaf on the face's low side (reversed with `flip`).
  void finest_face_segments(const Lattice& x, int64_t len, int a, bool flip, std::vector<Segment>& out) {
    const int u = (a + 1) % 3, v = (a + 2) % 3;
    Lattice c[4] = {x, x, x, x};
    c[1][u] += len;
    c[2][u] += len;
    c[2][v] += len;
    c[3][v] += len;
    std::vector<Lattice> cycle;
    edge_points(c[0], c[1], u, cycle);
    edge_points(c[1], c[2], v, cycle);
    edge_points(c[2], c[3], u, cycle);
    edge_points(c[3], c[0], v, cycle);
    // The crossings, in boundary order: (vertex, inside -> outside?)
    std::vector<std::pair<int, bool>> cross;
    const size_t m = cycle.size();
    for (size_t k = 0; k < m; ++k) {
      const Lattice &p = cycle[k], &q = cycle[(k + 1) % m];
      const bool ip = f_.inside(p), iq = f_.inside(q);
      if (ip != iq) cross.push_back({vertices_.get(p, q), ip});
    }
    // Crossings alternate between the two kinds: join each exit to the one before
    const size_t nc = cross.size();
    for (size_t k = 0; k < nc; ++k) {
      if (!cross[k].second) continue;
      const int from = cross[k].first, to = cross[(k + nc - 1) % nc].first;
      out.push_back(flip ? Segment{to, from} : Segment{from, to});
    }
  }

  // The segments on face 2a + s of leaf i, from the finest faces covering it
  // (the smaller leaves across it, if any), oriented for leaf i
  void leaf_face_segments(int i, int face, std::vector<Segment>& out) {
    const int a = face / 2, s = face % 2;
    const int nb = tree_.neighbor(i, face);
    if (nb < 0 || tree_.nodes[nb].child < 0) {
      Lattice x = tree_.nodes[i].x;
      x[a] += s * tree_.size(tree_.nodes[i]);
      finest_face_segments(x, tree_.size(tree_.nodes[i]), a, s == 0, out);
      return;
    }
    // The leaves under nb touching the face
    std::vector<int> stack{nb};
    while (!stack.empty()) {
      const int j = stack.back();
      stack.pop_back();
      const Node& n = tree_.nodes[j];
      if (n.child < 0) {
        Lattice x = n.x;
        x[a] += (1 - s) * tree_.size(n);
        finest_face_segments(x, tree_.size(n), a, s == 0, out);
        continue;
      }
      for (int k = 0; k < 8; ++k)
        if (((k >> a) & 1) == 1 - s) stack.push_back(n.child + k);
    }
  }

  // Chain a leaf's segments into closed polygons, counter-clockwise seen
  // from outside
  static void loops(const std::vector<Segment>& segments, std::vector<std::vector<int>>& out) {
    if (segments.empty()) return;
    std::unordered_map<int, int> next;
    for (const auto& sg : segments) next[sg[0]] = sg[1];
    std::unordered_map<int, bool> used;
    for (const auto& sg : segments) {
      if (used[sg[0]]) continue;
      std::vector<int> loop;
      for (int x = sg[0]; !used[x]; x = next.at(x)) {
        used[x] = true;
        loop.push_back(x);
      }
      std::reverse(loop.begin(), loop.end());  // the segments run clockwise
      out.push_back(std::move(loop));
    }
  }

  static double tri_area(const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 u{b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const Vec3 n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    return 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  }

  // Bit 2a + s: vertex id lies on face 2a + s of leaf i (its edge lies in
  // that face's plane; it lies on the leaf's boundary, so also in the face)
  int faces_of(int id, int i) const {
    const Node& n = tree_.nodes[i];
    const auto& e = vertices_.edge(id);
    int mask = 0;
    for (int a = 0; a < 3; ++a) {
      if (e[0][a] != e[1][a]) continue;
      if (e[0][a] == n.x[a]) mask |= 1 << (2 * a);
      if (e[0][a] == n.x[a] + tree_.size(n)) mask |= 1 << (2 * a + 1);
    }
    return mask;
  }

  // Triangulates polygon `poly` of leaf `leaf`, with minimal area (dynamic
  // programming over its diagonals), as in PoissonRecon, but using no
  // diagonal between two vertices on one face of the leaf. Two leaves'
  // triangles can only share an edge between vertices both leaves have,
  // i.e. on their common face (or edge): either a contour segment, used once
  // by each, or such a diagonal, which both leaves could otherwise pick,
  // making it non-manifold. If no triangulation avoids them, the polygon is
  // fanned around its centroid, a new vertex.
  void triangulate(const std::vector<int>& poly, int leaf) {
    const int n = int(poly.size());
    if (n < 3) return;
    if (n == 3) {
      add_triangle(mesh_, poly[0], poly[1], poly[2]);
      return;
    }
    std::vector<int> faces(n);
    for (int k = 0; k < n; ++k) faces[k] = faces_of(poly[k], leaf);
    // Diagonal (i, j), i < j: allowed unless its ends share a face; (0, n-1)
    // is a side of the polygon
    auto allowed = [&](int i, int j) { return (i == 0 && j == n - 1) || (faces[i] & faces[j]) == 0; };
    const double inf = std::numeric_limits<double>::infinity();
    const auto& V = mesh_.vertices;
    std::vector<double> cost(n * n, 0.0);
    std::vector<int> best(n * n, -1);
    for (int len = 2; len < n; ++len)
      for (int i = 0; i + len < n; ++i) {
        const int j = i + len;
        double c = inf;
        if (allowed(i, j))
          for (int k = i + 1; k < j; ++k) {
            const double ck = cost[i * n + k] + cost[k * n + j] + tri_area(V[poly[i]], V[poly[k]], V[poly[j]]);
            if (ck < c) {
              c = ck;
              best[i * n + j] = k;
            }
          }
        cost[i * n + j] = c;
      }
    if (cost[n - 1] == inf) {
      Vec3 c{0, 0, 0};
      for (int k = 0; k < n; ++k)
        for (int a = 0; a < 3; ++a) c[a] += V[poly[k]][a] / n;
      const int center = int(mesh_.vertices.size());
      mesh_.vertices.push_back(c);
      for (int k = 0; k < n; ++k) add_triangle(mesh_, poly[k], poly[(k + 1) % n], center);
      return;
    }
    std::vector<std::pair<int, int>> stack{{0, n - 1}};
    while (!stack.empty()) {
      const auto [i, j] = stack.back();
      stack.pop_back();
      if (j - i < 2) continue;
      const int k = best[i * n + j];
      add_triangle(mesh_, poly[i], poly[k], poly[j]);
      stack.push_back({i, k});
      stack.push_back({k, j});
    }
  }
};

}  // namespace

Mesh psr(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt) {
  return PSR(subdivide, f, opt).run();
}

}  // namespace detail
}  // namespace omc
