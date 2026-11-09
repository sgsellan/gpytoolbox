// Marching cubes on an adaptive octree.
//
// Standalone (C++17, standard library only): copy octree_mc.h, octree_mc.cpp
// and transvoxel_tables.h into any project.
//
// The octree is refined wherever `subdivide` says so, down to max_depth, then
// balanced so that leaves sharing a face differ by at most one level (this
// may split a few more cells than `subdivide` asked for). Marching cubes runs
// on every leaf. The cracks where a leaf meets four leaves one level smaller
// are closed with the Transvoxel algorithm (Lengyel 2010): a transition cell
// on the larger leaf's side of that face joins its contour to the smaller
// leaves'. The mesh is watertight. Transition cells get a width of 1/4 of the
// larger leaf: its vertices near the face are moved inward, along the
// surface, which avoids flat folds at the face.
//
// Convention: f < 0 is inside, f >= 0 outside; triangles are oriented with
// their normals pointing outside (counter-clockwise seen from outside).
#pragma once

#include <array>
#include <functional>
#include <vector>

namespace omc {

using Vec3 = std::array<double, 3>;

// An octree cell: the box [lo, hi], at `depth` (0 for the root).
struct Cell {
  Vec3 lo, hi;
  int depth;
};

// Both callbacks are batched, so that they can be vectorized (e.g. numpy
// from Python): marching_cubes calls them a few times on many inputs rather
// than once per input. See pointwise / per_cell below for one-at-a-time
// functions.
//
// The field whose zero level set is extracted (e.g. an SDF): its values at
// the given points. It is called once with the octree's corners, once per
// binary search step, and once for the normals of the vertices moved by
// transition cells.
using ScalarField = std::function<std::vector<double>(const std::vector<Vec3>& points)>;
// Which of the given cells to split into 8 children. It is called once per
// depth, with all the cells of that depth (only those shallower than
// max_depth).
using SubdivideFn = std::function<std::vector<bool>(const std::vector<Cell>& cells)>;

// Adapters from functions of one point / one cell
inline ScalarField pointwise(std::function<double(const Vec3&)> f) {
  return [f = std::move(f)](const std::vector<Vec3>& points) {
    std::vector<double> values(points.size());
    for (size_t i = 0; i < points.size(); ++i) values[i] = f(points[i]);
    return values;
  };
}
inline SubdivideFn per_cell(std::function<bool(const Cell&)> subdivide) {
  return [s = std::move(subdivide)](const std::vector<Cell>& cells) {
    std::vector<bool> split(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) split[i] = s(cells[i]);
    return split;
  };
}

// How a vertex is placed on a cell edge whose endpoints have opposite signs:
// linear interpolation of the endpoint values (no extra evaluations), or
// binary search on f (binary_search_steps evaluations; the vertex is the
// midpoint of the final bracket).
enum class Interpolation { Linear, BinarySearch };

struct Options {
  Vec3 lo{-1, -1, -1}, hi{1, 1, 1};  // the root cell
  int max_depth = 6;                 // 0..20
  Interpolation interpolation = Interpolation::Linear;
  int binary_search_steps = 10;
};

struct Mesh {
  std::vector<Vec3> vertices;
  std::vector<std::array<int, 3>> triangles;
};

Mesh marching_cubes(const SubdivideFn& subdivide, const ScalarField& f,
                    const Options& options = {});

}  // namespace omc
