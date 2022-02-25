#include <npe.h>
#include <igl/copyleft/cgal/mesh_boolean.h>

npe_function(mesh_boolean)
npe_arg(v, dense_double)
npe_arg(f, dense_int)
npe_begin_code()
    Eigen::MatrixXd V(v);
    Eigen::MatrixXi F(f);
//    remesh_botsch(V, F, target, i, feature, project);
    return std::make_tuple(npe::move(V), npe::move(V));
npe_end_code()