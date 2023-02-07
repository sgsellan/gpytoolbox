#include "remesher/remesh_botsch.h"
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

void binding_remesh_botsch(py::module& m) {
    m.def("_remesh_botsch_cpp_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, int i, double h, EigenDRef<VectorXi> feature, bool project)
        {
            Eigen::MatrixXd V(v);
            Eigen::MatrixXi F(f);
            Eigen::VectorXi FEATURE(feature);
            remesh_botsch(V, F, h, i, FEATURE, project);
            return std::make_tuple(V, F);
        });
    
}