#include "outside_points_from_rasterization.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
#include <iostream>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_outside_points_from_rasterization(py::module& m) {
    m.def("_outside_points_from_rasterization_cpp_impl",[](EigenDRef<MatrixXd> _sdf_points,
                         EigenDRef<MatrixXd> _sphere_radii,
                         int rng_seed,
                         int res,
                         double tol,
                         bool narrow_band,
                         bool parallel,
                         bool force_cpu,
                         bool verbose)
        {
            Eigen::MatrixXd sdf_points(_sdf_points), sphere_radii(_sphere_radii);
            Eigen::MatrixXd outside_points;
            const int dim = sdf_points.cols();
            if(dim==2) {
                outside_points_from_rasterization<2>(sdf_points, sphere_radii,
                    rng_seed, res, tol,
                    narrow_band,
                    parallel,
                    force_cpu,
                    verbose,
                    outside_points);
            } else if(dim==3) {
                outside_points_from_rasterization<3>(sdf_points, sphere_radii,
                    rng_seed, res, tol,
                    narrow_band,
                    parallel,
                    force_cpu,
                    verbose,
                    outside_points);
            }
            return outside_points;
        });
    
}
