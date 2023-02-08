#include "swept_volume/swept_volume.h"
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

void binding_swept_volume(py::module& m) {
    m.def("_swept_volume_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, EigenDRef<MatrixXd> transformations, double eps, int num_seeds, bool verbose)
        {
            Eigen::MatrixXd V(v);
            Eigen::MatrixXi F(f);
            Eigen::MatrixXd U;
            Eigen::MatrixXi G;
            swept_volume(V, F,transformations, eps, num_seeds, verbose, U,G);
            return std::make_tuple(U,G);
        });
    
}