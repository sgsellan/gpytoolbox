// #include <npe.h>
// #include <pybind11/stl.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
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

void binding_booleans(py::module& m) {
    m.def("_mesh_union_cpp_impl",[](EigenDRef<MatrixXd> va,
                         EigenDRef<MatrixXi> fa, EigenDRef<MatrixXd> vb,
                         EigenDRef<MatrixXi> fb)
        {
            Eigen::MatrixXd VA(va);
            Eigen::MatrixXi FA(fa);
            Eigen::MatrixXd VB(vb);
            Eigen::MatrixXi FB(fb);
            Eigen::MatrixXd VC;
            Eigen::MatrixXi FC;
            igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,igl::MESH_BOOLEAN_TYPE_UNION,VC,FC);
            return std::make_tuple(VC, FC);
        });
    m.def("_mesh_intersection_cpp_impl",[](EigenDRef<MatrixXd> va,
                         EigenDRef<MatrixXi> fa, EigenDRef<MatrixXd> vb,
                         EigenDRef<MatrixXi> fb)
        {
            Eigen::MatrixXd VA(va);
            Eigen::MatrixXi FA(fa);
            Eigen::MatrixXd VB(vb);
            Eigen::MatrixXi FB(fb);
            Eigen::MatrixXd VC;
            Eigen::MatrixXi FC;
            igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,igl::MESH_BOOLEAN_TYPE_INTERSECT,VC,FC);
            return std::make_tuple(VC, FC);
        });
    m.def("_mesh_difference_cpp_impl",[](EigenDRef<MatrixXd> va,
                         EigenDRef<MatrixXi> fa, EigenDRef<MatrixXd> vb,
                         EigenDRef<MatrixXi> fb)
        {
            Eigen::MatrixXd VA(va);
            Eigen::MatrixXi FA(fa);
            Eigen::MatrixXd VB(vb);
            Eigen::MatrixXi FB(fb);
            Eigen::MatrixXd VC;
            Eigen::MatrixXi FC;
            igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,igl::MESH_BOOLEAN_TYPE_MINUS,VC,FC);
            return std::make_tuple(VC, FC);
        });
    m.def("_do_meshes_intersect_cpp_impl",[](EigenDRef<MatrixXd> va,
                         EigenDRef<MatrixXi> fa, EigenDRef<MatrixXd> vb,
                         EigenDRef<MatrixXi> fb)
        {
            Eigen::MatrixXd VA(va);
            Eigen::MatrixXi FA(fa);
            Eigen::MatrixXd VB(vb);
            Eigen::MatrixXi FB(fb);
            Eigen::MatrixXd VVAB;
            Eigen::MatrixXi FFAB,IF;
            Eigen::VectorXi JAB,IMAB;
            igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
            params.detect_only = true;
            params.first_only = true;
            igl::copyleft::cgal::intersect_other(VA,FA,VB,FB,params,IF,VVAB,FFAB,JAB,IMAB);
            return IF;
        });
    
}

