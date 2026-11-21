// Method::Transvoxel: marching cubes on the leaves of a balanced octree, the
// cracks between leaves of different sizes closed with transition cells
// (Lengyel, "Voxel-Based Terrain for Real-Time Virtual Simulations", 2010).
#include "octree_mc_common.h"
#include "transvoxel_tables.h"

namespace omc {
namespace detail {
namespace {

namespace tv = transvoxel;

// Transition cells take this fraction of the larger leaf (Lengyel's width)
constexpr double kTransitionWidth = 0.25;

class Transvoxel {
 public:
  Transvoxel(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt)
      : subdivide_(subdivide), f_(f), tree_(opt) {}

  Mesh run() {
    tree_.build(subdivide_);
    balance();
    tree_.collect_leaves();
    f_.evaluate_corners(tree_);
    // The mesh's topology: vertices are only identified (by their edges) here
    for (int i : tree_.leaves) {
      const int mask = transition_mask(i);
      regular_cell(i, mask);
      for (int face = 0; face < 6; ++face)
        if (mask >> face & 1) transition_cell(i, face);
    }
    mesh_.vertices = vertices_.place(tree_, f_);
    apply_shifts();
    return std::move(mesh_);
  }

 private:
  const SubdivideFn& subdivide_;
  Field f_;
  Octree tree_;
  EdgeVertices vertices_;
  // A leaf with transition faces (bits as in transition_mask) and one of its
  // vertices, to move by the transition width
  struct ShiftRequest {
    int vertex, leaf, transitions;
  };
  std::vector<ShiftRequest> shift_requests_;
  Mesh mesh_;

  // Split leaves until
  // - leaves sharing a face differ by at most one level, and
  // - no edge has a checkerboard: a leaf whose two face neighbors along one
  //   of its edges are smaller, while the leaf diagonally across the edge is
  //   not. There the transition cells of the two large leaves would all
  //   meet on one mesh edge, making it non-manifold; the leaf is split.
  void balance() {
    auto& nodes = tree_.nodes;
    for (bool changed = true; changed;) {
      changed = false;
      const int n = int(nodes.size());
      for (int i = 0; i < n; ++i) {
        if (nodes[i].child >= 0) continue;
        const int d = nodes[i].depth;
        for (int face = 0; face < 6; ++face) {
          const int nb = tree_.neighbor(i, face);
          if (nb >= 0 && nodes[nb].child < 0 && nodes[nb].depth < d - 1) {
            tree_.split(nb);
            changed = true;
          }
        }
        if (nodes[i].child >= 0 || d >= tree_.max_depth()) continue;
        auto finer = [&](int j) { return j >= 0 && nodes[j].depth == d && nodes[j].child >= 0; };
        bool checkerboard = false;
        for (int f1 = 0; f1 < 6 && !checkerboard; ++f1)
          for (int f2 = f1 + 1; f2 < 6 && !checkerboard; ++f2) {
            if (f1 / 2 == f2 / 2 || !finer(tree_.neighbor(i, f1)) || !finer(tree_.neighbor(i, f2))) continue;
            const int diag = tree_.across(i, f1, f2);
            checkerboard = diag >= 0 && nodes[diag].child < 0;
          }
        if (checkerboard) {
          tree_.split(i);
          changed = true;
        }
      }
    }
  }

  // Bit 2a + s: the leaf's face 2a + s borders leaves one level smaller
  int transition_mask(int i) const {
    int mask = 0;
    for (int face = 0; face < 6; ++face) {
      const int nb = tree_.neighbor(i, face);
      if (nb >= 0 && tree_.nodes[nb].depth == tree_.nodes[i].depth && tree_.nodes[nb].child >= 0)
        mask |= 1 << face;
    }
    return mask;
  }

  // Marching cubes on leaf i (Lengyel's regular cell tables)
  void regular_cell(int i, int transitions) {
    const Node& n = tree_.nodes[i];
    Lattice p[8];
    int code = 0;
    for (int k = 0; k < 8; ++k) {
      for (int a = 0; a < 3; ++a) p[k][a] = n.x[a] + ((k >> a) & 1) * tree_.size(n);
      if (f_.inside(p[k])) code |= 1 << k;
    }
    if (code == 0 || code == 255) return;
    const tv::RegularCellData& data = tv::regularCellData[tv::regularCellClass[code]];
    int ids[12];
    for (long k = 0; k < data.GetVertexCount(); ++k) {
      const int e = tv::regularVertexData[code][k] & 0xFF;
      ids[k] = vertices_.get(p[e >> 4], p[e & 0xF]);
      if (transitions) shift_requests_.push_back({ids[k], i, transitions});
    }
    for (long t = 0; t < data.GetTriangleCount(); ++t)
      add_triangle(mesh_, ids[data.vertexIndex[3 * t]], ids[data.vertexIndex[3 * t + 1]],
                   ids[data.vertexIndex[3 * t + 2]]);
  }

  // The transition cell on face 2a + s of leaf i, joining the leaf's contour
  // on that face (the half-resolution samples 9..C, its corners) to the
  // contours of the 4 smaller leaves (the full-resolution samples 0..8)
  void transition_cell(int i, int face) {
    const Node& n = tree_.nodes[i];
    const int a = face / 2, s = face % 2;
    // A frame (u, v) of the face with u x v pointing out of the leaf
    int u = (a + 1) % 3, v = (a + 2) % 3;
    if (s == 0) std::swap(u, v);
    const int64_t h = tree_.size(n) / 2;
    Lattice p[13];
    for (int k = 0; k < 9; ++k) {
      p[k] = n.x;
      p[k][a] += s * tree_.size(n);
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
      if (f_.inside(p[k])) code |= bit[k];
    if (code == 0 || code == 511) return;
    const unsigned cls = tv::transitionCellClass[code];
    const tv::TransitionCellData& data = tv::transitionCellData[cls & 0x7F];
    int ids[12];
    for (long k = 0; k < data.GetVertexCount(); ++k) {
      const int e = tv::transitionVertexData[code][k] & 0xFF;
      ids[k] = vertices_.get(p[e >> 4], p[e & 0xF]);
    }
    // Bit 0x80 marks the cases whose winding is reversed. In our frame (u x v
    // pointing out of the larger leaf) the table's winding faces inside, so
    // the unmarked cases are the ones to flip (checked by the tests)
    for (long t = 0; t < data.GetTriangleCount(); ++t)
      add_triangle(mesh_, ids[data.vertexIndex[3 * t]], ids[data.vertexIndex[3 * t + 1]],
                   ids[data.vertexIndex[3 * t + 2]], (cls & 0x80) == 0);
  }

  // Each leaf with transition faces shrinks away from them by the transition
  // width, moving its vertices; a vertex shared by several such leaves takes,
  // on each axis, the largest move. The move is projected onto the surface's
  // tangent plane (normal from central differences of f, one evaluation for
  // all the vertices) so the vertex stays on the surface, with two
  // constraints:
  // - a vertex on a root face stays on it: the axes across the faces it lies
  //   on (known from its edge's lattice endpoints, exactly, whatever the
  //   bounds) are left out of both the gradient and the move, so it moves
  //   along the level set within the face;
  // - no vertex leaves the root cell: the projected move is scaled down,
  //   as a whole to keep it tangent, until it stays inside.
  void apply_shifts() {
    const Options& opt = tree_.options();
    const int64_t R = tree_.resolution();
    std::vector<Vec3> shift(mesh_.vertices.size(), Vec3{0, 0, 0});
    for (const ShiftRequest& r : shift_requests_) {
      const Node& n = tree_.nodes[r.leaf];
      Lattice hi = n.x;
      for (auto& c : hi) c += tree_.size(n);
      const Vec3 lo = tree_.world(n.x), up = tree_.world(hi), &x = mesh_.vertices[r.vertex];
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

    // The vertices to move, the axes they may move along, and the probes of
    // the central differences along those axes
    struct Move {
      int vertex;
      bool free[3];
      int first_probe;
    };
    std::vector<Move> moves;
    std::vector<Vec3> probes;
    double h = 1e-4;
    for (int a = 0; a < 3; ++a) h = std::min(h, 1e-4 * std::ldexp(opt.hi[a] - opt.lo[a], -tree_.max_depth()));
    for (size_t id = 0; id < shift.size(); ++id) {
      const auto& e = vertices_.edge(int(id));
      Move m{int(id), {true, true, true}, int(probes.size())};
      bool any = false;
      for (int a = 0; a < 3; ++a) {
        m.free[a] = !(e[0][a] == e[1][a] && (e[0][a] == 0 || e[0][a] == R));
        if (!m.free[a]) shift[id][a] = 0;
        any |= shift[id][a] != 0;
      }
      if (!any) continue;
      for (int a = 0; a < 3; ++a) {
        if (!m.free[a]) continue;
        for (double sign : {1.0, -1.0}) {
          Vec3 p = mesh_.vertices[id];
          p[a] += sign * h;
          probes.push_back(p);
        }
      }
      moves.push_back(m);
    }
    const std::vector<double> values = f_(probes);

    for (const Move& m : moves) {
      const Vec3& d = shift[m.vertex];
      Vec3& x = mesh_.vertices[m.vertex];
      Vec3 g{0, 0, 0};
      double g2 = 0;
      for (int a = 0, k = m.first_probe; a < 3; ++a) {
        if (!m.free[a]) continue;
        g[a] = values[k] - values[k + 1];
        g2 += g[a] * g[a];
        k += 2;
      }
      const double dn = g2 > 0 ? (d[0] * g[0] + d[1] * g[1] + d[2] * g[2]) / g2 : 0.0;
      Vec3 move;
      double scale = 1;
      for (int a = 0; a < 3; ++a) {
        move[a] = m.free[a] ? d[a] - dn * g[a] : 0.0;
        if (move[a] > 0) scale = std::min(scale, (opt.hi[a] - x[a]) / move[a]);
        if (move[a] < 0) scale = std::min(scale, (opt.lo[a] - x[a]) / move[a]);
      }
      scale = std::max(scale, 0.0);
      for (int a = 0; a < 3; ++a) x[a] += scale * move[a];
    }
  }
};

}  // namespace

Mesh transvoxel(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt) {
  return Transvoxel(subdivide, f, opt).run();
}

}  // namespace detail
}  // namespace omc
