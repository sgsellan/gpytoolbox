#include "octree_mc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "transvoxel_tables.h"

namespace omc {
namespace {

namespace tv = transvoxel;

// Transition cells take this fraction of the larger leaf (Lengyel's width)
constexpr double kTransitionWidth = 0.25;

// Positions are integer points of the 2^max_depth lattice over the root cell
using Lattice = std::array<int64_t, 3>;

uint64_t pack(const Lattice& p) {
  return uint64_t(p[0]) | uint64_t(p[1]) << 21 | uint64_t(p[2]) << 42;
}

struct Node {
  Lattice x;  // the low corner
  int depth;
  int child = -1;  // the first of 8 consecutive children, or -1 for a leaf
};

// A mesh vertex is identified by the lattice edge it lies on
struct EdgeKey {
  uint64_t a, b;
  bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};
struct EdgeHash {
  size_t operator()(const EdgeKey& k) const {
    return size_t(k.a * 0x9E3779B97F4A7C15ull ^ (k.b + 0x632BE59BD9B4E019ull + (k.a << 6) + (k.a >> 2)));
  }
};

class Mesher {
 public:
  Mesher(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt)
      : subdivide_(subdivide), f_(f), opt_(opt), D_(opt.max_depth) {}

  Mesh run() {
    build();
    balance();
    collect_leaves(0);
    evaluate_corners();
    // The mesh's topology: vertices are only identified (by their edges) here
    for (int i : leaves_) {
      const int mask = transition_mask(i);
      regular_cell(i, mask);
      for (int face = 0; face < 6; ++face)
        if (mask >> face & 1) transition_cell(i, face);
    }
    place_vertices();
    apply_shifts();
    return std::move(mesh_);
  }

 private:
  const SubdivideFn& subdivide_;
  const ScalarField& f_;
  const Options& opt_;
  const int D_;
  std::vector<Node> nodes_;
  std::vector<int> leaves_;
  std::unordered_map<uint64_t, double> values_;  // f at the leaves' corners
  std::unordered_map<EdgeKey, int, EdgeHash> vertex_of_;
  std::vector<std::array<Lattice, 2>> edges_;  // per mesh vertex: its edge
  // A leaf with transition faces (bits as in transition_mask) and one of its
  // vertices, to move by the transition width
  struct ShiftRequest {
    int vertex, leaf, transitions;
  };
  std::vector<ShiftRequest> shift_requests_;
  Mesh mesh_;

  int64_t size(const Node& n) const { return int64_t(1) << (D_ - n.depth); }

  Vec3 world(const Lattice& p) const {
    Vec3 w;
    for (int a = 0; a < 3; ++a)
      w[a] = opt_.lo[a] + (opt_.hi[a] - opt_.lo[a]) * std::ldexp(double(p[a]), -D_);
    return w;
  }

  // f at many points, checking what the callback returns
  std::vector<double> eval(const std::vector<Vec3>& points) const {
    if (points.empty()) return {};
    std::vector<double> values = f_(points);
    if (values.size() != points.size())
      throw std::runtime_error("the field returned " + std::to_string(values.size()) + " values for " +
                               std::to_string(points.size()) + " points");
    return values;
  }

  double value(const Lattice& p) const { return values_.at(pack(p)); }

  // f at the corners of every leaf, in one call (transition cells only use
  // these: their samples are the corners of the smaller leaves)
  void evaluate_corners() {
    std::vector<uint64_t> keys;
    std::vector<Vec3> points;
    for (int i : leaves_) {
      const Node& n = nodes_[i];
      for (int k = 0; k < 8; ++k) {
        Lattice p;
        for (int a = 0; a < 3; ++a) p[a] = n.x[a] + ((k >> a) & 1) * size(n);
        if (values_.emplace(pack(p), 0.0).second) {
          keys.push_back(pack(p));
          points.push_back(world(p));
        }
      }
    }
    const std::vector<double> values = eval(points);
    for (size_t k = 0; k < keys.size(); ++k) values_[keys[k]] = values[k];
  }

  void split(int i) {
    const int first = int(nodes_.size());
    nodes_[i].child = first;
    const Node n = nodes_[i];
    const int64_t h = size(n) / 2;
    for (int k = 0; k < 8; ++k) {
      Node c{n.x, n.depth + 1};
      for (int a = 0; a < 3; ++a) c.x[a] += ((k >> a) & 1) * h;
      nodes_.push_back(c);
    }
  }

  // Top-down, one depth at a time: subdivide sees all the cells of a depth
  void build() {
    nodes_.assign(1, Node{{0, 0, 0}, 0});
    std::vector<int> level{0};
    for (int d = 0; d < D_ && !level.empty(); ++d) {
      std::vector<Cell> cells;
      cells.reserve(level.size());
      for (int i : level) {
        const Node& n = nodes_[i];
        Lattice hi = n.x;
        for (auto& c : hi) c += size(n);
        cells.push_back(Cell{world(n.x), world(hi), d});
      }
      const std::vector<bool> split_it = subdivide_(cells);
      if (split_it.size() != cells.size())
        throw std::runtime_error("subdivide returned " + std::to_string(split_it.size()) + " answers for " +
                                 std::to_string(cells.size()) + " cells");
      std::vector<int> next;
      for (size_t k = 0; k < level.size(); ++k) {
        if (!split_it[k]) continue;
        split(level[k]);
        for (int c = 0; c < 8; ++c) next.push_back(nodes_[level[k]].child + c);
      }
      level.swap(next);
    }
  }

  // The node holding point q of the doubled lattice, no deeper than max_depth
  int locate(const Lattice& q, int max_depth) const {
    int i = 0;
    while (nodes_[i].child >= 0 && nodes_[i].depth < max_depth) {
      const Node& n = nodes_[i];
      int k = 0;
      for (int a = 0; a < 3; ++a) k |= int(q[a] >= 2 * n.x[a] + size(n)) << a;
      i = n.child + k;
    }
    return i;
  }

  // The node of the same depth (or the coarser leaf) across face 2a + s of
  // node i, or -1 at the root cell's boundary
  int neighbor(int i, int face) const { return across(i, face, -1); }

  // The same across two faces (2a + s, 2b + t, a != b), i.e. the node
  // diagonally across the edge they share; face2 = -1 for a single face
  int across(int i, int face1, int face2) const {
    const Node& n = nodes_[i];
    Lattice q;
    for (int b = 0; b < 3; ++b) q[b] = 2 * n.x[b] + size(n);  // the center
    for (int face : {face1, face2}) {
      if (face < 0) continue;
      const int a = face / 2, s = face % 2;
      q[a] = s ? 2 * (n.x[a] + size(n)) + 1 : 2 * n.x[a] - 1;
      if (q[a] < 0 || q[a] > (int64_t(2) << D_)) return -1;
    }
    return locate(q, n.depth);
  }

  // Split leaves until
  // - leaves sharing a face differ by at most one level, and
  // - no edge has a checkerboard: a leaf whose two face neighbors along one
  //   of its edges are smaller, while the leaf diagonally across the edge is
  //   not. There the transition cells of the two large leaves would all
  //   meet on one mesh edge, making it non-manifold; the leaf is split.
  void balance() {
    for (bool changed = true; changed;) {
      changed = false;
      const int n = int(nodes_.size());
      for (int i = 0; i < n; ++i) {
        if (nodes_[i].child >= 0) continue;
        const int d = nodes_[i].depth;
        for (int face = 0; face < 6; ++face) {
          const int nb = neighbor(i, face);
          if (nb >= 0 && nodes_[nb].child < 0 && nodes_[nb].depth < d - 1) {
            split(nb);
            changed = true;
          }
        }
        if (nodes_[i].child >= 0 || d >= D_) continue;
        auto finer = [&](int j) { return j >= 0 && nodes_[j].depth == d && nodes_[j].child >= 0; };
        bool checkerboard = false;
        for (int f1 = 0; f1 < 6 && !checkerboard; ++f1)
          for (int f2 = f1 + 1; f2 < 6 && !checkerboard; ++f2) {
            if (f1 / 2 == f2 / 2 || !finer(neighbor(i, f1)) || !finer(neighbor(i, f2))) continue;
            const int diag = across(i, f1, f2);
            checkerboard = diag >= 0 && nodes_[diag].child < 0;
          }
        if (checkerboard) {
          split(i);
          changed = true;
        }
      }
    }
  }

  void collect_leaves(int i) {
    if (nodes_[i].child < 0) {
      leaves_.push_back(i);
      return;
    }
    for (int k = 0; k < 8; ++k) collect_leaves(nodes_[i].child + k);
  }

  // Bit 2a + s: the leaf's face 2a + s borders leaves one level smaller
  int transition_mask(int i) const {
    int mask = 0;
    for (int face = 0; face < 6; ++face) {
      const int nb = neighbor(i, face);
      if (nb >= 0 && nodes_[nb].depth == nodes_[i].depth && nodes_[nb].child >= 0)
        mask |= 1 << face;
    }
    return mask;
  }

  // The vertex on the lattice edge [p, q] (whose values have opposite
  // signs); it is placed later, by place_vertices
  int vertex(const Lattice& p, const Lattice& q) {
    uint64_t kp = pack(p), kq = pack(q);
    const bool swap = kp > kq;
    if (swap) std::swap(kp, kq);
    auto [it, added] = vertex_of_.emplace(EdgeKey{kp, kq}, int(edges_.size()));
    if (added) edges_.push_back(swap ? std::array<Lattice, 2>{q, p} : std::array<Lattice, 2>{p, q});
    return it->second;
  }

  // Every vertex on its edge: linear interpolation, or binary search with
  // one evaluation of f per step for all the vertices
  void place_vertices() {
    const size_t n = edges_.size();
    mesh_.vertices.resize(n);
    std::vector<Vec3> x0(n), x1(n);  // the brackets: x0 on the first endpoint's side
    std::vector<char> first_inside(n);
    for (size_t k = 0; k < n; ++k) {
      const Vec3 pa = world(edges_[k][0]), pb = world(edges_[k][1]);
      const double va = value(edges_[k][0]), vb = value(edges_[k][1]);
      if (opt_.interpolation == Interpolation::Linear) {
        const double t = std::clamp(va / (va - vb), 0.0, 1.0);
        for (int a = 0; a < 3; ++a) mesh_.vertices[k][a] = pa[a] + t * (pb[a] - pa[a]);
      }
      x0[k] = pa;
      x1[k] = pb;
      first_inside[k] = va < 0;
    }
    if (opt_.interpolation == Interpolation::Linear) return;
    std::vector<Vec3> mid(n);
    auto midpoints = [&]() {
      for (size_t k = 0; k < n; ++k)
        for (int a = 0; a < 3; ++a) mid[k][a] = 0.5 * (x0[k][a] + x1[k][a]);
    };
    for (int s = 0; s < opt_.binary_search_steps; ++s) {
      midpoints();
      const std::vector<double> values = eval(mid);
      for (size_t k = 0; k < n; ++k) {
        if ((values[k] < 0) == bool(first_inside[k])) x0[k] = mid[k];
        else x1[k] = mid[k];
      }
    }
    midpoints();
    mesh_.vertices = mid;
  }

  void triangle(int a, int b, int c, bool flip) {
    if (a == b || b == c || c == a) return;
    if (flip) std::swap(b, c);
    mesh_.triangles.push_back({a, b, c});
  }

  // Marching cubes on leaf i (Lengyel's regular cell tables)
  void regular_cell(int i, int transitions) {
    const Node& n = nodes_[i];
    Lattice p[8];
    int code = 0;
    for (int k = 0; k < 8; ++k) {
      for (int a = 0; a < 3; ++a) p[k][a] = n.x[a] + ((k >> a) & 1) * size(n);
      if (value(p[k]) < 0) code |= 1 << k;
    }
    if (code == 0 || code == 255) return;
    const tv::RegularCellData& data = tv::regularCellData[tv::regularCellClass[code]];
    int ids[12];
    for (long k = 0; k < data.GetVertexCount(); ++k) {
      const int e = tv::regularVertexData[code][k] & 0xFF;
      ids[k] = vertex(p[e >> 4], p[e & 0xF]);
      if (transitions) shift_requests_.push_back({ids[k], i, transitions});
    }
    for (long t = 0; t < data.GetTriangleCount(); ++t)
      triangle(ids[data.vertexIndex[3 * t]], ids[data.vertexIndex[3 * t + 1]],
               ids[data.vertexIndex[3 * t + 2]], false);
  }

  // The transition cell on face 2a + s of leaf i, joining the leaf's contour
  // on that face (the half-resolution samples 9..C, its corners) to the
  // contours of the 4 smaller leaves (the full-resolution samples 0..8)
  void transition_cell(int i, int face) {
    const Node& n = nodes_[i];
    const int a = face / 2, s = face % 2;
    // A frame (u, v) of the face with u x v pointing out of the leaf
    int u = (a + 1) % 3, v = (a + 2) % 3;
    if (s == 0) std::swap(u, v);
    const int64_t h = size(n) / 2;
    Lattice p[13];
    for (int k = 0; k < 9; ++k) {
      p[k] = n.x;
      p[k][a] += s * size(n);
      p[k][u] += (k % 3) * h;
      p[k][v] += (k / 3) * h;
    }
    p[9] = p[0];
    p[10] = p[2];
    p[11] = p[6];
    p[12] = p[8];
    static const int bit[9] = {0x01, 0x02, 0x04, 0x80, 0x100, 0x08, 0x40, 0x20, 0x10};
    int code = 0;
    for (int k = 0; k < 9; ++k)
      if (value(p[k]) < 0) code |= bit[k];
    if (code == 0 || code == 511) return;
    const unsigned cls = tv::transitionCellClass[code];
    const tv::TransitionCellData& data = tv::transitionCellData[cls & 0x7F];
    int ids[12];
    for (long k = 0; k < data.GetVertexCount(); ++k) {
      const int e = tv::transitionVertexData[code][k] & 0xFF;
      ids[k] = vertex(p[e >> 4], p[e & 0xF]);
    }
    // Bit 0x80 marks the cases whose winding is reversed. In our frame (u x v
    // pointing out of the larger leaf) the table's winding faces inside, so
    // the unmarked cases are the ones to flip (checked by the tests)
    for (long t = 0; t < data.GetTriangleCount(); ++t)
      triangle(ids[data.vertexIndex[3 * t]], ids[data.vertexIndex[3 * t + 1]],
               ids[data.vertexIndex[3 * t + 2]], (cls & 0x80) == 0);
  }

  // Each leaf with transition faces shrinks away from them by the
  // transition width, moving its vertices; a vertex shared by several such
  // leaves takes, on each axis, the largest move. The move is projected onto
  // the surface's tangent plane (normal from central differences of f, one
  // evaluation for all the vertices) so the vertex stays on the surface
  void apply_shifts() {
    std::vector<Vec3> shift(mesh_.vertices.size(), Vec3{0, 0, 0});
    for (const ShiftRequest& r : shift_requests_) {
      const Node& n = nodes_[r.leaf];
      Lattice hi = n.x;
      for (auto& c : hi) c += size(n);
      const Vec3 lo = world(n.x), up = world(hi), &x = mesh_.vertices[r.vertex];
      for (int a = 0; a < 3; ++a) {
        const double len = up[a] - lo[a];
        const double wl = (r.transitions >> (2 * a) & 1) ? kTransitionWidth * len : 0.0;
        const double wh = (r.transitions >> (2 * a + 1) & 1) ? kTransitionWidth * len : 0.0;
        if (wl == 0 && wh == 0) continue;
        const double t = x[a] - lo[a];
        const double d = wl + t * (len - wl - wh) / len - t;
        if (std::abs(d) > std::abs(shift[r.vertex][a])) shift[r.vertex][a] = d;
      }
    }
    std::vector<int> moved;
    for (size_t id = 0; id < shift.size(); ++id)
      if (shift[id][0] != 0 || shift[id][1] != 0 || shift[id][2] != 0) moved.push_back(int(id));
    double h = 1e-4;
    for (int a = 0; a < 3; ++a) h = std::min(h, 1e-4 * std::ldexp(opt_.hi[a] - opt_.lo[a], -D_));
    std::vector<Vec3> probes;
    probes.reserve(6 * moved.size());
    for (int id : moved)
      for (int a = 0; a < 3; ++a)
        for (double sign : {1.0, -1.0}) {
          Vec3 p = mesh_.vertices[id];
          p[a] += sign * h;
          probes.push_back(p);
        }
    const std::vector<double> values = eval(probes);
    for (size_t k = 0; k < moved.size(); ++k) {
      const Vec3& d = shift[moved[k]];
      Vec3& x = mesh_.vertices[moved[k]];
      Vec3 g;
      double g2 = 0;
      for (int a = 0; a < 3; ++a) {
        g[a] = values[6 * k + 2 * a] - values[6 * k + 2 * a + 1];
        g2 += g[a] * g[a];
      }
      const double dn = g2 > 0 ? (d[0] * g[0] + d[1] * g[1] + d[2] * g[2]) / g2 : 0.0;
      // ...except across the root cell's faces: vertices on them stay there
      for (int a = 0; a < 3; ++a)
        if (x[a] != opt_.lo[a] && x[a] != opt_.hi[a]) x[a] += d[a] - dn * g[a];
    }
  }
};

}  // namespace

Mesh marching_cubes(const SubdivideFn& subdivide, const ScalarField& f,
                    const Options& opt) {
  if (opt.max_depth < 0 || opt.max_depth > 20)
    throw std::invalid_argument("max_depth must be in 0..20");
  if (opt.interpolation == Interpolation::BinarySearch && opt.binary_search_steps < 0)
    throw std::invalid_argument("binary_search_steps must be >= 0");
  for (int a = 0; a < 3; ++a)
    if (!(opt.lo[a] < opt.hi[a])) throw std::invalid_argument("need lo < hi");
  return Mesher(subdivide, f, opt).run();
}

}  // namespace omc
