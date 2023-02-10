#include "tinyply/tinyply_wrappers.h"
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

void binding_write_ply(py::module& m) {
    m.def("_write_ply_cpp_impl",[](std::string filename, EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, EigenDRef<MatrixXd> n,
                         EigenDRef<MatrixXd> c)
        {
            return write_ply(filename, v, f, n, c);
        });
}