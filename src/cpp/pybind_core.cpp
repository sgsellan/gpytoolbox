
#include <pybind11/stl.h>
#include "read_obj.h"
#include "write_obj.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily


PYBIND11_MODULE(gpytoolbox_bindings, m) {
    m.def("read_obj_pybind",[](std::string filename,
     bool return_UV, bool return_N)
        {
            Eigen::MatrixXd V, UV, N;
            Eigen::MatrixXi F, Ft, Fn;

            int err = read_obj(filename, return_UV, return_N,
                V, F, UV, Ft, N, Fn);
            return std::make_tuple(err, V, F, UV, Ft, N, Fn);
        });

    m.def("write_obj_pybind",[](std::string filename, EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, EigenDRef<MatrixXd> uv,
                         EigenDRef<MatrixXi> ft, EigenDRef<MatrixXd> n,
                         EigenDRef<MatrixXi> fn)
        {
            return write_obj(filename, v, f, uv, ft, n, fn);
        });
}

