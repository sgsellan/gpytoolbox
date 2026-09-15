#include "triangulate_polygon.h"
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

void binding_triangulate_polygon(py::module& m) {
    // triangulate_polygon throws a std::invalid_argument for a polygon CDT
    // refuses, which pybind11 turns into a python ValueError
    m.def("_triangulate_polygon_cpp_impl",[](EigenDRef<MatrixXd> v,
                EigenDRef<MatrixXi> f, double a, double q, bool steiner) {
            Eigen::MatrixXd V(v);
            Eigen::MatrixXi F(f);
            Eigen::MatrixXd V2;
            Eigen::MatrixXi F2;
            triangulate_polygon(V,F,a,q,steiner,V2,F2);
            return std::make_tuple(V2,F2);
        });
}
