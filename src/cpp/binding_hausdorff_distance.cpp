#include <npe.h>
#include <pybind11/stl.h>
#include <igl/hausdorff.h>

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(_hausdorff_distance_cpp_impl)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(ut, dense_double)
npe_arg(gt, dense_int)
npe_begin_code()
    double s;
    // Eigen::MatrixXd V(vt);
    // Eigen::MatrixXi F(ft);
    // Eigen::MatrixXd Q(qt);
    igl::hausdorff(vt,ft,ut,gt,s);
    return s;
npe_end_code()

