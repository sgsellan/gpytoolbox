#include <npe.h>
#include <pybind11/stl.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>


npe_function(mesh_union)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    // Eigen::MatrixXd va(va);
    // Eigen::MatrixXi fa(fa);
    // Eigen::MatrixXd vb(vb);
    // Eigen::MatrixXi fb(fb);
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::copyleft::cgal::mesh_boolean(va,fa,vb,fb,igl::MESH_BOOLEAN_TYPE_UNION,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()

npe_function(mesh_intersection)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    // Eigen::MatrixXd va(va);
    // Eigen::MatrixXi fa(fa);
    // Eigen::MatrixXd vb(vb);
    // Eigen::MatrixXi fb(fb);
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::copyleft::cgal::mesh_boolean(va,fa,vb,fb,igl::MESH_BOOLEAN_TYPE_INTERSECT,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()

npe_function(mesh_difference)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    // Eigen::MatrixXd va(va);
    // Eigen::MatrixXi fa(fa);
    // Eigen::MatrixXd vb(vb);
    // Eigen::MatrixXi fb(fb);
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::copyleft::cgal::mesh_boolean(va,fa,vb,fb,igl::MESH_BOOLEAN_TYPE_MINUS,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()

npe_function(do_meshes_intersect)
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

