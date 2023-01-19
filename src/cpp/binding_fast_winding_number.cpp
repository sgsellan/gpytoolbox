#include <npe.h>
#include <pybind11/stl.h>
#include <igl/fast_winding_number.h>

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(_fast_winding_number_cpp_impl)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(qt, dense_double)
npe_begin_code()
    Eigen::VectorXd S;
    // Eigen::MatrixXd V(vt);
    // Eigen::MatrixXi F(ft);
    // Eigen::MatrixXd Q(qt);
    igl::fast_winding_number(vt,ft,qt,S);
    return npe::move(S);
npe_end_code()

