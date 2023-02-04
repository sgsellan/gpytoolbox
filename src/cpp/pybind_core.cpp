
#include <pybind11/stl.h>
#include "read_obj.h"
#include "write_obj.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>

#include <igl/opengl/glfw/Viewer.h>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

//forward declare all other binding functions outside of core
void bind_viewer(py::module& m);

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
    bind_viewer(m);

    m.def("help", [&]() {printf("hi"); });
    //wrap the viewer class. For now we assume that a user won't have multiple cameras, but may have multiple meshes. 
    //Note that we do not wrap the data class yet, users directly set data quantities by specifying which mesh they're talking about
    // with a mesh id.
    //py::class_<igl::opengl::glfw::Viewer>(m, "viewer")
    //    //set mesh
    //    .def(py::init<>())
    //    //.def("set_mesh", [&](iglv::Viewer& v, MatrixXd& V, MatrixXi& F) {
    //    //	v.data().set_mesh(V, F); 
    //    //	})
    //    ////.def("set_mesh", [&](Viewer& v, MatrixXd& V, MatrixXi& F, int i) {
    //    ////		v.data_list[i].set_mesh(V, F);
    //    ////	})
    //    //.def("launch", [&](iglv::Viewer& v) 
    //    //	{v.launch(); })
    //    ;

    //bind viewer

}

