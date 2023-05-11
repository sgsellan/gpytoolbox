#include "per_face_prin_curvature.h"
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

void binding_per_face_prin_curvature(py::module& m) {
    m.def("_per_face_prin_curvature_cpp_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f)
        {
            Eigen::MatrixXd PD1,PD2;
            Eigen::VectorXd PC1, PC2;
            per_face_prin_curvature(v,f,PD1,PD2,PC1,PC2);
            return std::make_tuple(PD1,PD2,PC1,PC2);
        });
    
}

