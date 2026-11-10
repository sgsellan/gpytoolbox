#include "octree_mc.h"

#include <stdexcept>

#include "octree_mc_common.h"

namespace omc {

Mesh marching_cubes(const SubdivideFn& subdivide, const ScalarField& f, const Options& opt) {
  if (opt.max_depth < 0 || opt.max_depth > 20) throw std::invalid_argument("max_depth must be in 0..20");
  if (opt.interpolation == Interpolation::BinarySearch && opt.binary_search_steps < 0)
    throw std::invalid_argument("binary_search_steps must be >= 0");
  for (int a = 0; a < 3; ++a)
    if (!(opt.lo[a] < opt.hi[a])) throw std::invalid_argument("need lo < hi");
  if (!subdivide || !f) throw std::invalid_argument("subdivide and f must be set");
  return opt.method == Method::PSR ? detail::psr(subdivide, f, opt) : detail::transvoxel(subdivide, f, opt);
}

}  // namespace omc
