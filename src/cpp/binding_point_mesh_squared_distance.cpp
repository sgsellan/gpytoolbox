#include <npe.h>
#include <pybind11/stl.h>
#include <igl/point_mesh_squared_distance.h>

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(_point_mesh_squared_distance_cpp_impl)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(qt, dense_double)
npe_begin_code()
    Eigen::VectorXd sqrD;
    Eigen::MatrixXd C;
    Eigen::VectorXi I;
    // Eigen::MatrixXd V(vt);
    // Eigen::MatrixXi F(ft);
    // Eigen::MatrixXd Q(qt);
    igl::point_mesh_squared_distance(qt,vt,ft,sqrD,I,C);
    return std::make_tuple(npe::move(sqrD),npe::move(I),npe::move(C));
npe_end_code()

