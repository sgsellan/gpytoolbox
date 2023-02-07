#include "ray_mesh_intersect_aabb.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_ray_mesh_intersect(py::module& m) {
    m.def("_ray_mesh_intersect_cpp_impl",[](EigenDRef<MatrixXd> cam_pos,EigenDRef<MatrixXd> cam_dir, EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f)
        {
            Eigen::VectorXi ids;
            Eigen::VectorXd ts;
            Eigen::MatrixXd lambdas;
            ray_mesh_intersect_aabb(cam_pos, cam_dir, v, f, ts, ids, lambdas);
            return std::make_tuple(ts,ids,lambdas);
        });
    
}

