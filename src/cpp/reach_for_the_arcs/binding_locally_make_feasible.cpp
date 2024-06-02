#include "locally_make_feasible.h"
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

void binding_locally_make_feasible(py::module& m) {
    m.def("_locally_make_feasible_cpp_impl",[](EigenDRef<MatrixXd> _sdf_points,
                         EigenDRef<MatrixXd> _sdf_data,
                         EigenDRef<MatrixXd> _outside_points,
                         EigenDRef<MatrixXi> _batch,
                         int rng_seed,
                         const int n_local_searches,
                         const int local_search_iters,
                         const double tol,
                         const double clamp_value,
                         bool parallel,
                         bool verbose)
        {
            Eigen::MatrixXd sdf_points(_sdf_points), sdf_data(_sdf_data),
            outside_points(_outside_points);
            Eigen::VectorXi batch(_batch);
            Eigen::MatrixXd cloud_points, cloud_normals;
            Eigen::MatrixXi feasible;
            const int dim = sdf_points.cols();
            if(dim==2) {
                locally_make_feasible<2>(sdf_points, sdf_data, outside_points,
                    batch,
                    rng_seed, n_local_searches, local_search_iters,
                    tol, clamp_value, parallel, verbose,
                    cloud_points, cloud_normals, feasible);
            } else if(dim==3) {
                locally_make_feasible<3>(sdf_points, sdf_data, outside_points,
                    batch,
                    rng_seed, n_local_searches, local_search_iters,
                    tol, clamp_value, parallel, verbose,
                    cloud_points, cloud_normals, feasible);
            }
            return std::make_tuple(cloud_points, cloud_normals, feasible);
        });
    
}
