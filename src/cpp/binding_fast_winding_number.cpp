// #include <npe.h>
#include <igl/fast_winding_number.h>
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

void binding_fast_winding_number(py::module& m) {
    m.def("_fast_winding_number_cpp_impl",[](EigenDRef<MatrixXd> vt,
                         EigenDRef<MatrixXi> ft, EigenDRef<MatrixXd> qt)
        {
            Eigen::VectorXd S;
            igl::fast_winding_number(vt,ft,qt,S);
            return S;
        });
}
