#include <npe.h>
#include <pybind11/stl.h>
#include <igl/offset_surface.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/Hit.h>
#include <igl/decimate.h>
#include <upper_envelope.h>
#include <ray_mesh_intersect_aabb.h>
#include <in_element_aabb.h>

npe_function(mesh_union)
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

npe_function(mesh_intersection)
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

npe_function(mesh_difference)
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

npe_function(offset_surface)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(iso, double)
npe_arg(grid_size, int)
npe_begin_code()
    Eigen::MatrixXd VA(va);
    Eigen::MatrixXi FA(fa);
    Eigen::MatrixXd VB;
    Eigen::MatrixXi FB;
    Eigen::MatrixXd GV;
    Eigen::RowVector3i side;
    Eigen::VectorXd S;
    igl::offset_surface(VA,FA,iso,grid_size,igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,VB,FB,GV,side,S);
    return std::make_tuple(npe::move(VB),npe::move(FB));
npe_end_code()



npe_function(ray_mesh_intersect)
npe_arg(cam_pos, dense_double)
npe_arg(cam_dir, dense_double)
npe_arg(v, dense_double)
npe_arg(f, dense_int)
npe_begin_code()
    Eigen::MatrixXd CAM_POS(cam_pos);
    Eigen::MatrixXd CAM_DIR(cam_dir);
    Eigen::MatrixXd V(v);
    Eigen::MatrixXi F(f);
    Eigen::VectorXi ids;
    Eigen::VectorXd ts;
    Eigen::MatrixXd lambdas;
    ray_mesh_intersect_aabb(CAM_POS, CAM_DIR, V, F, ts, ids, lambdas);
    return std::make_tuple(npe::move(ts),npe::move(ids),npe::move(lambdas));
npe_end_code()



// void upper_envelope(const Eigen::MatrixXd VT, const Eigen::MatrixXi FT, const Eigen::MatrixXd DT, Eigen::MatrixXd & UT, Eigen::MatrixXi & GT, Eigen::MatrixXd LT);
npe_function(upper_envelope)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(dt, dense_double)
npe_begin_code()
    Eigen::MatrixXd VT(vt);
    Eigen::MatrixXi FT(ft);
    Eigen::MatrixXd DT(dt);
    Eigen::MatrixXd UT;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> LT;
    Eigen::MatrixXi GT;
    upper_envelope(VT,FT,DT,UT,GT,LT);
    return std::make_tuple(npe::move(UT),npe::move(GT),npe::move(LT));
npe_end_code()

npe_function(in_element_aabb)
npe_arg(queries, dense_double)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_begin_code()
    Eigen::MatrixXd P(queries);
    Eigen::MatrixXd V(vt);
    Eigen::MatrixXi F(ft);
    Eigen::VectorXi I;
    in_element_aabb(P,V,F,I);
    return npe::move(I);
npe_end_code()

// remove_duplicate_vertices(V,Q,np.amin(W)/100)
npe_function(remove_duplicate_vertices)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(eps, double)
npe_begin_code()
    Eigen::MatrixXd V(vt);
    Eigen::MatrixXi F(ft);
    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    Eigen::VectorXi SVI, SVJ;
    igl::remove_duplicate_vertices(V,F,eps,SV,SVI,SVJ,SF);
    return std::make_tuple(npe::move(SV),npe::move(SVI),npe::move(SVJ),npe::move(SF));
npe_end_code()


// decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(decimate)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(num_faces, int)
npe_begin_code()
    Eigen::MatrixXd V(vt);
    Eigen::MatrixXi F(ft);
    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    Eigen::VectorXi J, I;
    igl::decimate(V,F,num_faces,SV,SF,I,J);
    return std::make_tuple(npe::move(SV),npe::move(SF),npe::move(I),npe::move(J));
npe_end_code()

