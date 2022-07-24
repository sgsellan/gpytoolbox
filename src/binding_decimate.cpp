#include <npe.h>
#include <pybind11/stl.h>
#include <igl/decimate.h>

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(decimate)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(num_faces, int)
npe_begin_code()
    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    Eigen::VectorXi J, I;
    igl::decimate(vt,ft,num_faces,SV,SF,I,J);
    return std::make_tuple(npe::move(SV),npe::move(SF),npe::move(I),npe::move(J));
npe_end_code()

