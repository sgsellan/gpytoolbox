#include <igl/point_mesh_squared_distance.h>
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

void binding_point_mesh_squared_distance(py::module& m) {
    m.def("_point_mesh_squared_distance_cpp_impl",[](EigenDRef<MatrixXd> vt,
                         EigenDRef<MatrixXi> ft, EigenDRef<MatrixXd> qt)
        {
            Eigen::VectorXd sqrD;
            Eigen::MatrixXd C;
            Eigen::VectorXi I;
            igl::point_mesh_squared_distance(qt,vt,ft,sqrD,I,C);
            return std::make_tuple(sqrD,I,C);
        });
}