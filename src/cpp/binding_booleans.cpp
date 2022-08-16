#include <npe.h>
#include <pybind11/stl.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>


npe_function(_mesh_union_cpp_impl)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    Eigen::MatrixXd VA(va);
    Eigen::MatrixXi FA(fa);
    Eigen::MatrixXd VB(vb);
    Eigen::MatrixXi FB(fb);
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,igl::MESH_BOOLEAN_TYPE_UNION,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()

npe_function(_mesh_intersection_cpp_impl)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    Eigen::MatrixXd VA(va);
    Eigen::MatrixXi FA(fa);
    Eigen::MatrixXd VB(vb);
    Eigen::MatrixXi FB(fb);
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,igl::MESH_BOOLEAN_TYPE_INTERSECT,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()

npe_function(_mesh_difference_cpp_impl)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    Eigen::MatrixXd VA(va);
    Eigen::MatrixXi FA(fa);
    Eigen::MatrixXd VB(vb);
    Eigen::MatrixXi FB(fb);
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,igl::MESH_BOOLEAN_TYPE_MINUS,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()

npe_function(_do_meshes_intersect_cpp_impl)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
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
    return std::make_tuple(npe::move(IF));
npe_end_code()

