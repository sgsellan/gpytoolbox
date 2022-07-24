#include <npe.h>
#include <pybind11/stl.h>
#include <ray_mesh_intersect_aabb.h>

// npe_function(offset_surface)
// npe_arg(va, dense_double)
// npe_arg(fa, dense_int)
// npe_arg(iso, double)
// npe_arg(grid_size, int)
// npe_begin_code()
//     Eigen::MatrixXd VA(va);
//     Eigen::MatrixXi FA(fa);
//     Eigen::MatrixXd VB;
//     Eigen::MatrixXi FB;
//     Eigen::MatrixXd GV;
//     Eigen::RowVector3i side;
//     Eigen::VectorXd S;
//     igl::offset_surface(VA,FA,iso,grid_size,igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,VB,FB,GV,side,S);
//     return std::make_tuple(npe::move(VB),npe::move(FB));
// npe_end_code()



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



// // void upper_envelope(const Eigen::MatrixXd VT, const Eigen::MatrixXi FT, const Eigen::MatrixXd DT, Eigen::MatrixXd & UT, Eigen::MatrixXi & GT, Eigen::MatrixXd LT);
// npe_function(upper_envelope)
// npe_arg(vt, dense_double)
// npe_arg(ft, dense_int)
// npe_arg(dt, dense_double)
// npe_begin_code()
//     Eigen::MatrixXd VT(vt);
//     Eigen::MatrixXi FT(ft);
//     Eigen::MatrixXd DT(dt);
//     Eigen::MatrixXd UT;
//     Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> LT;
//     Eigen::MatrixXi GT;
//     upper_envelope(VT,FT,DT,UT,GT,LT);
//     return std::make_tuple(npe::move(UT),npe::move(GT),npe::move(LT));
// npe_end_code()

// npe_function(in_element_aabb)
// npe_arg(queries, dense_double)
// npe_arg(vt, dense_double)
// npe_arg(ft, dense_int)
// npe_begin_code()
//     Eigen::MatrixXd P(queries);
//     Eigen::MatrixXd V(vt);
//     Eigen::MatrixXi F(ft);
//     Eigen::VectorXi I;
//     in_element_aabb(P,V,F,I);
//     return npe::move(I);
// npe_end_code()

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
// npe_function(decimate)
// npe_arg(vt, dense_double)
// npe_arg(ft, dense_int)
// npe_arg(num_faces, int)
// npe_begin_code()
//     Eigen::MatrixXd V(vt);
//     Eigen::MatrixXi F(ft);
//     Eigen::MatrixXd SV;
//     Eigen::MatrixXi SF;
//     Eigen::VectorXi J, I;
//     igl::decimate(V,F,num_faces,SV,SF,I,J);
//     return std::make_tuple(npe::move(SV),npe::move(SF),npe::move(I),npe::move(J));
// npe_end_code()


// npe_function(mqwf)
// npe_arg(A, sparse_double)
// npe_arg(B, dense_double)
// npe_arg(known, dense_int)
// npe_arg(Y, npe_matches(B))
// npe_arg(Aeq, npe_matches(A))
// npe_arg(Beq, npe_matches(B))
// npe_begin_code()
//     Eigen::SparseMatrix<double> A_copy(A);
//     Eigen::SparseMatrix<double> Aeq_copy(Aeq);
//     Eigen::MatrixXd B_copy(B);
//     Eigen::MatrixXd Y_copy(Y);
//     Eigen::MatrixXd Beq_copy(Beq);
//     Eigen::MatrixXd sol;
//     bool is_A_pd = true;
//     bool ok = igl::min_quad_with_fixed(A_copy, B_copy, known, Y_copy, Aeq_copy, Beq_copy, is_A_pd, sol);
//     return npe::move(sol);
// npe_end_code()




// // Remesher
// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(t, dense_double)
// npe_arg(i, int)
// npe_arg(ft, dense_int)
// npe_arg(project, bool)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     Eigen::VectorXd target(t);
//     Eigen::VectorXi feature(ft);
//     remesh_botsch(V, F, target, i, feature, project);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(t, dense_double)
// npe_arg(i, int)
// npe_arg(ft, dense_int)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     Eigen::VectorXd target(t);
//     Eigen::VectorXi feature(ft);
//     remesh_botsch(V, F, target, i, feature, false);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(target, dense_double)
// npe_arg(i, int)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     Eigen::VectorXd t(target);
//     remesh_botsch(V, F, t, i);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(target, dense_double)
// npe_arg(i, int)
// npe_arg(project, bool)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     Eigen::VectorXd t(target);
//     remesh_botsch(V, F, t, i, project);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(target, dense_double)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     Eigen::VectorXd t(target);
//     remesh_botsch(V, F, t);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(i, int)
// npe_arg(h, double)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     remesh_botsch(V, F, h, i);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(i, int)
// npe_arg(h, double)
// npe_arg(project, bool)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     remesh_botsch(V, F, h, i, project);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(h, double)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     remesh_botsch(V, F, h);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(remesh_botsch)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     remesh_botsch(V, F);
//     return std::make_tuple(npe::move(V), npe::move(F));
// npe_end_code()

// npe_function(_read_obj_cpp_impl)
// npe_arg(file, std::string)
// npe_arg(return_UV, bool)
// npe_arg(return_N, bool)
// npe_begin_code()
//     Eigen::MatrixXd V, UV, N;
//     Eigen::MatrixXi F, Ft, Fn;
//     int err = read_obj(file, return_UV, return_N,
//         V, F, UV, Ft, N, Fn);
//     return std::make_tuple(err, V, F, UV, Ft, N, Fn);
// npe_end_code()

// npe_function(_write_obj_cpp_impl)
// npe_arg(file, std::string)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(uv, dense_double)
// npe_arg(ft, dense_int)
// npe_arg(n, dense_double)
// npe_arg(fn, dense_int)
// npe_begin_code()
//     Eigen::MatrixXd V(v), UV(uv), N(n);
//     Eigen::MatrixXi F(f), Ft(ft), Fn(fn);
//     return write_obj(file, V, F, UV, Ft, N, Fn);
// npe_end_code()
// ///