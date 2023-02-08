#include "write_obj.h"
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

void binding_write_obj(py::module& m) {
    m.def("_write_obj_cpp_impl",[](std::string filename, EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, EigenDRef<MatrixXd> uv,
                         EigenDRef<MatrixXi> ft, EigenDRef<MatrixXd> n,
                         EigenDRef<MatrixXi> fn)
        {
            return write_obj(filename, v, f, uv, ft, n, fn);
        });
}