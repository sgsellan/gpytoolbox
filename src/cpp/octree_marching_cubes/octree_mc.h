// Marching cubes on an adaptive octree.
//
// Standalone (C++17, standard library only): copy the files of src/ into any
// project (octree_mc.h is the interface).
//
// The octree is refined wherever `subdivide` says so, down to max_depth.
// Marching cubes runs on every leaf; the cracks between leaves of different
// sizes are closed by one of two methods (Options::method), both giving
// watertight meshes:
// - PSR (default), the extraction of Poisson Surface Reconstruction (Kazhdan
//   et al. 2007): the octree is used as built; each leaf takes its contour on
//   a face from the smaller leaves across it, and its vertices from the
//   smallest edges, and triangulates its contour polygons with minimal area.
//   More accurate for the same number of triangles, and needs fewer.
// - Transvoxel (Lengyel 2010): the octree is balanced (leaves sharing a face
//   differ by at most one level, which may split a few more cells than
//   `subdivide` asked for), and a transition cell on the larger leaf's side
//   of each such face joins its contour to the smaller leaves'. Transition
//   cells get a width of 1/4 of the larger leaf: its vertices near the face
//   are moved inward, along the surface (which can move vertices slightly
//   off curved surfaces). Keeps the surface area better.
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
// binary search step, and (Transvoxel) once for the normals of the vertices
// moved by transition cells.
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

// How the cracks between leaves of different sizes are closed (see above)
enum class Method { Transvoxel, PSR };

struct Options {
  Vec3 lo{-1, -1, -1}, hi{1, 1, 1};  // the root cell
  int max_depth = 6;                 // 0..20
  Interpolation interpolation = Interpolation::Linear;
  int binary_search_steps = 10;
  Method method = Method::PSR;
};

struct Mesh {
  std::vector<Vec3> vertices;
  std::vector<std::array<int, 3>> triangles;
};

Mesh marching_cubes(const SubdivideFn& subdivide, const ScalarField& f,
                    const Options& options = {});

}  // namespace omc
