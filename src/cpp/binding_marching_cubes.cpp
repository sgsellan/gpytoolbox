#include <npe.h>
#include <pybind11/stl.h>
#include <igl/marching_cubes.h>

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(_marching_cubes_cpp_impl)
npe_arg(s, dense_double)
npe_arg(GV, dense_double)
npe_arg(nx, int)
npe_arg(nz, int)
npe_arg(ny, int)
npe_arg(iso, double)
npe_begin_code()
    Eigen::VectorXd S(s);
    // Eigen::MatrixXd GV;
    Eigen::MatrixXi F;
    Eigen::MatrixXd V;
    igl::marching_cubes(S, GV, nx, ny, nz, iso, V, F);
    return std::make_tuple(npe::move(V),npe::move(F));
npe_end_code()

