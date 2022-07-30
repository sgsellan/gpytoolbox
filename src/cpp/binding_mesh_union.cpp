#include <npe.h>
#include <pybind11/stl.h>
#include "mesh_union.h"

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
npe_function(_mesh_union_cpp_impl)
npe_arg(va, dense_double)
npe_arg(fa, dense_int)
npe_arg(vb, dense_double)
npe_arg(fb, dense_int)
npe_begin_code()
    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    mesh_union(va,fa,vb,fb,igl::MESH_BOOLEAN_TYPE_UNION,VC,FC);
    return std::make_tuple(npe::move(VC), npe::move(FC));
npe_end_code()
