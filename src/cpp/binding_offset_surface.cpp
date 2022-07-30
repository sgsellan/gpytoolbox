#include <npe.h>
#include <pybind11/stl.h>
#include "igl/offset_surface.h"

npe_function(_offset_surface_cpp_impl)
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