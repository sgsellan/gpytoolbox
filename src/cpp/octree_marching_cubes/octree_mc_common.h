// Internal to octree_mc: the parts shared by both crack-free methods (the
// octree, the field at its corners, and the vertices on its edges), so that
// they only differ in how they make the mesh crack-free.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "octree_mc.h"

namespace omc {
namespace detail {

// Positions are integer points of the 2^max_depth lattice over the root cell
using Lattice = std::array<int64_t, 3>;

inline uint64_t pack(const Lattice& p) {
  return uint64_t(p[0]) | uint64_t(p[1]) << 21 | uint64_t(p[2]) << 42;
}

struct Node {
  Lattice x;  // the low corner
  int depth;
  int child = -1;  // the first of 8 consecutive children, or -1 for a leaf
};

class Octree {
 public:
  explicit Octree(const Options& opt) : opt_(opt), D_(opt.max_depth) {}

  const Options& options() const { return opt_; }
  int max_depth() const { return D_; }
  // The lattice coordinate of the root cell's high faces
  int64_t resolution() const { return int64_t(1) << D_; }
  int64_t size(const Node& n) const { return int64_t(1) << (D_ - n.depth); }

  // The root cell's faces map exactly to lo and hi (lo + (hi - lo) * 1 may
  // round past hi)
  Vec3 world(const Lattice& p) const {
    Vec3 w;
    for (int a = 0; a < 3; ++a)
      w[a] = p[a] == resolution() ? opt_.hi[a]
                                  : opt_.lo[a] + (opt_.hi[a] - opt_.lo[a]) * std::ldexp(double(p[a]), -D_);
    return w;
  }

  // Top-down, one depth at a time: subdivide sees all the cells of a depth
  void build(const SubdivideFn& subdivide) {
    nodes.assign(1, Node{{0, 0, 0}, 0});
    std::vector<int> level{0};
    for (int d = 0; d < D_ && !level.empty(); ++d) {
      std::vector<Cell> cells;
      cells.reserve(level.size());
      for (int i : level) {
        const Node& n = nodes[i];
        Lattice hi = n.x;
        for (auto& c : hi) c += size(n);
        cells.push_back(Cell{world(n.x), world(hi), d});
      }
      const std::vector<bool> split_it = subdivide(cells);
      if (split_it.size() != cells.size())
        throw std::runtime_error("subdivide returned " + std::to_string(split_it.size()) + " answers for " +
                                 std::to_string(cells.size()) + " cells");
      std::vector<int> next;
      for (size_t k = 0; k < level.size(); ++k) {
        if (!split_it[k]) continue;
        split(level[k]);
        for (int c = 0; c < 8; ++c) next.push_back(nodes[level[k]].child + c);
      }
      level.swap(next);
    }
  }

  void split(int i) {
    nodes[i].child = int(nodes.size());
    const Node n = nodes[i];
    const int64_t h = size(n) / 2;
    for (int k = 0; k < 8; ++k) {
      Node c{n.x, n.depth + 1};
      for (int a = 0; a < 3; ++a) c.x[a] += ((k >> a) & 1) * h;
      nodes.push_back(c);
    }
  }

  void collect_leaves() {
    leaves.clear();
    std::vector<int> stack{0};
    while (!stack.empty()) {
      const int i = stack.back();
      stack.pop_back();
      if (nodes[i].child < 0) leaves.push_back(i);
      else
        for (int k = 7; k >= 0; --k) stack.push_back(nodes[i].child + k);
    }
  }

  // The node holding point q of the doubled lattice, no deeper than max_depth
  int locate(const Lattice& q, int max_depth) const {
    int i = 0;
    while (nodes[i].child >= 0 && nodes[i].depth < max_depth) {
      const Node& n = nodes[i];
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
    const Node& n = nodes[i];
    Lattice q;
    for (int b = 0; b < 3; ++b) q[b] = 2 * n.x[b] + size(n);  // the center
    for (int face : {face1, face2}) {
      if (face < 0) continue;
      const int a = face / 2, s = face % 2;
      q[a] = s ? 2 * (n.x[a] + size(n)) + 1 : 2 * n.x[a] - 1;
      if (q[a] < 0 || q[a] > 2 * resolution()) return -1;
    }
    return locate(q, n.depth);
  }

  std::vector<Node> nodes;
  std::vector<int> leaves;

 private:
  const Options& opt_;
  const int D_;
};

// The field: batched evaluation (checking what the callback returns), and its
// values at the leaves' corners, the only lattice points both methods sample
class Field {
 public:
  explicit Field(const ScalarField& f) : f_(f) {}

  std::vector<double> operator()(const std::vector<Vec3>& points) const {
    if (points.empty()) return {};
    std::vector<double> values = f_(points);
    if (values.size() != points.size())
      throw std::runtime_error("the field returned " + std::to_string(values.size()) + " values for " +
                               std::to_string(points.size()) + " points");
    return values;
  }

  // f at the corners of every leaf, in one call
  void evaluate_corners(const Octree& tree) {
    std::vector<uint64_t> keys;
    std::vector<Vec3> points;
    for (int i : tree.leaves) {
      const Node& n = tree.nodes[i];
      for (int k = 0; k < 8; ++k) {
        Lattice p;
        for (int a = 0; a < 3; ++a) p[a] = n.x[a] + ((k >> a) & 1) * tree.size(n);
        if (values_.emplace(pack(p), 0.0).second) {
          keys.push_back(pack(p));
          points.push_back(tree.world(p));
        }
      }
    }
    const std::vector<double> values = (*this)(points);
    for (size_t k = 0; k < keys.size(); ++k) values_[keys[k]] = values[k];
  }

  double at(const Lattice& p) const { return values_.at(pack(p)); }
  bool inside(const Lattice& p) const { return at(p) < 0; }

 private:
  const ScalarField& f_;
  std::unordered_map<uint64_t, double> values_;
};

// The mesh vertices, each identified by the lattice edge it lies on (whose
// endpoints have opposite signs): registered first, placed all at once
class EdgeVertices {
 public:
  int get(const Lattice& p, const Lattice& q) {
    uint64_t kp = pack(p), kq = pack(q);
    const bool swap = kp > kq;
    if (swap) std::swap(kp, kq);
    auto [it, added] = id_.emplace(Key{kp, kq}, int(edges_.size()));
    if (added) edges_.push_back(swap ? std::array<Lattice, 2>{q, p} : std::array<Lattice, 2>{p, q});
    return it->second;
  }

  size_t size() const { return edges_.size(); }
  const std::array<Lattice, 2>& edge(int id) const { return edges_[id]; }

  // Linear interpolation, or binary search with one evaluation of f per step
  // for all the vertices (the vertex is the midpoint of the final bracket)
  std::vector<Vec3> place(const Octree& tree, const Field& f) const {
    const Options& opt = tree.options();
    const size_t n = edges_.size();
    std::vector<Vec3> x(n), x0(n), x1(n);  // the brackets: x0 on the first endpoint's side
    std::vector<char> first_inside(n);
    for (size_t k = 0; k < n; ++k) {
      const Vec3 pa = tree.world(edges_[k][0]), pb = tree.world(edges_[k][1]);
      const double va = f.at(edges_[k][0]), vb = f.at(edges_[k][1]);
      const double t = std::clamp(va / (va - vb), 0.0, 1.0);
      for (int a = 0; a < 3; ++a) x[k][a] = pa[a] + t * (pb[a] - pa[a]);
      x0[k] = pa;
      x1[k] = pb;
      first_inside[k] = va < 0;
    }
    if (opt.interpolation == Interpolation::Linear) return x;
    auto midpoints = [&]() {
      for (size_t k = 0; k < n; ++k)
        for (int a = 0; a < 3; ++a) x[k][a] = 0.5 * (x0[k][a] + x1[k][a]);
    };
    for (int s = 0; s < opt.binary_search_steps; ++s) {
      midpoints();
      const std::vector<double> values = f(x);
      for (size_t k = 0; k < n; ++k) {
        if ((values[k] < 0) == bool(first_inside[k])) x0[k] = x[k];
        else x1[k] = x[k];
      }
    }
    midpoints();
    return x;
  }

 private:
  struct Key {
    uint64_t a, b;
    bool operator==(const Key& o) const { return a == o.a && b == o.b; }
  };
  struct Hash {
    size_t operator()(const Key& k) const {
      return size_t(k.a * 0x9E3779B97F4A7C15ull ^ (k.b + 0x632BE59BD9B4E019ull + (k.a << 6) + (k.a >> 2)));
    }
  };
  std::unordered_map<Key, int, Hash> id_;
  std::vector<std::array<Lattice, 2>> edges_;
};

// Appends a triangle, skipping degenerate ones (a repeated vertex)
inline void add_triangle(Mesh& mesh, int a, int b, int c, bool flip = false) {
  if (a == b || b == c || c == a) return;
  if (flip) std::swap(b, c);
  mesh.triangles.push_back({a, b, c});
}

// The two methods
Mesh transvoxel(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt);
Mesh psr(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt);

}  // namespace detail
}  // namespace omc
