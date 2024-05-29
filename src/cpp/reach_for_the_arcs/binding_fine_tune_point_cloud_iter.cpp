#include "fine_tune_point_cloud_iter.h"
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

void binding_fine_tune_point_cloud_iter(py::module& m) {
    m.def("_fine_tune_point_cloud_iter_cpp_impl",[](EigenDRef<MatrixXd> _sdf_points,
                         EigenDRef<MatrixXd> _sdf_data,
                         EigenDRef<MatrixXd> _reconstructed_V,
                         EigenDRef<MatrixXi> _reconstructed_F,
                         EigenDRef<MatrixXd> _cloud_points,
                         EigenDRef<MatrixXd> _cloud_normals,
                         EigenDRef<MatrixXi> _feasible,
                         EigenDRef<MatrixXi> _batch,
                         int max_points_per_sphere,
                         int rng_seed,
                         int n_local_searches,
                         int local_search_iters,
                         double local_search_t,
                         double tol,
                         double clamp_value,
                         bool parallel,
                         bool verbose)
        {
            Eigen::MatrixXd sdf_points(_sdf_points), sdf_data(_sdf_data),
            reconstructed_V(_reconstructed_V),
            cloud_points(_cloud_points), cloud_normals(_cloud_normals);
            Eigen::MatrixXi reconstructed_F(_reconstructed_F),
            feasible(_feasible);
            Eigen::VectorXi batch(_batch);
            const int dim = sdf_points.cols();
            if(dim==2) {
                fine_tune_point_cloud_iter<2>(sdf_points, sdf_data,
                    reconstructed_V, reconstructed_F,
                    cloud_points, cloud_normals, feasible,
                    batch,
                    max_points_per_sphere,
                    rng_seed, n_local_searches, local_search_iters,
                    local_search_t, tol, clamp_value, parallel, verbose);
            } else if(dim==3) {
                fine_tune_point_cloud_iter<3>(sdf_points, sdf_data,
                    reconstructed_V, reconstructed_F,
                    cloud_points, cloud_normals, feasible,
                    batch,
                    max_points_per_sphere,
                    rng_seed, n_local_searches, local_search_iters,
                    local_search_t, tol, clamp_value, parallel, verbose);
            }
            return std::make_tuple(cloud_points, cloud_normals, feasible);
        });
    
}
