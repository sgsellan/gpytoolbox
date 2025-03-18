#include <igl/decimate.h>
#include <igl/qslim.h>
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

void binding_decimate(py::module& m) {
    m.def("_decimate_cpp_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f,
                         int num_faces,
                         int method)
        {
            Eigen::MatrixXd SV;
            Eigen::MatrixXi SF;
            Eigen::VectorXi J, I;
    //         igl::decimate_pre_collapse_callback pre_collapse;
    //         igl::decimate_post_collapse_callback post_collapse;
    // igl::decimate_trivial_callbacks(pre_collapse,post_collapse);
            if(method==0) {
                igl::decimate(v,f,num_faces,
                    //This will be required when we bump the libigl version.
                    true,
                    SV,SF,I,J);
            } else if(method==1) {
                igl::qslim(v,f,num_faces,
                    //This will be required when we bump the libigl version.
                    true,
                    SV,SF,I,J);
            }
            return std::make_tuple(SV,SF,I,J);
        });
    
}
