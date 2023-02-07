#include "upper_envelope.h"
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

void binding_upper_envelope(py::module& m) {
    m.def("_upper_envelope_cpp_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f,EigenDRef<MatrixXd> d)
        {
            Eigen::MatrixXd VT(v);
            Eigen::MatrixXi FT(f);
            Eigen::MatrixXd DT(d);
            Eigen::MatrixXd UT;
            Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> LT;
            Eigen::MatrixXi GT;
            upper_envelope(VT,FT,DT,UT,GT,LT);
            return std::make_tuple(UT,GT,LT);
        });
    
}
