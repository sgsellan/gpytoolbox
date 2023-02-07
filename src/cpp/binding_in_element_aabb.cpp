#include "in_element_aabb.h"
#include <igl/hausdorff.h>
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

void binding_in_element_aabb(py::module& m) {
    m.def("_in_element_aabb_cpp_impl",[](EigenDRef<MatrixXd> qt, EigenDRef<MatrixXd> vt, EigenDRef<MatrixXi> ft)
        {
            Eigen::VectorXi I;
            in_element_aabb(qt,vt,ft,I);
            return I;
        });
}