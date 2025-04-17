#include <igl/decimate.h>
#include <igl/qslim.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
// debugging
#include <igl/writeOBJ.h>

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
            Eigen::MatrixXd V = v;
            Eigen::MatrixXi F = f;
            Eigen::MatrixXd SV;
            Eigen::MatrixXi SF;
            Eigen::VectorXi J, I;
            const bool block_intersections = false;
    //         igl::decimate_pre_collapse_callback pre_collapse;
    //         igl::decimate_post_collapse_callback post_collapse;
    // igl::decimate_trivial_callbacks(pre_collapse,post_collapse);
            if(method==0) {
                // std::cout << "Decimating with method 0" << std::endl;
                // igl::writeOBJ("decimate_input.obj",V,F);
                // std::cout << "Wrote input to decimate_input.obj" << std::endl;
                // std::cout << "Number of faces: " << num_faces << std::endl;
                igl::decimate(V,F,num_faces,
                    //This will be required when we bump the libigl version.
                    block_intersections,
                    SV,SF,I,J);
            } else if(method==1) {
                // std::cout << "Decimating with method 1" << std::endl;
                
                igl::qslim(V,F,num_faces,
                    //This will be required when we bump the libigl version.
                    block_intersections,
                    SV,SF,I,J);
            }
            return std::make_tuple(SV,SF,I,J);
        });
    
}