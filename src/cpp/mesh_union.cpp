#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include "mesh_union.h"

void mesh_union(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F1, const Eigen::MatrixXd& V2, const Eigen::MatrixXi& F2, Eigen::MatrixXd& V3, Eigen::MatrixXi& F1)
{
   igl::copyleft::cgal::mesh_boolean(V1,F1,V2,F2,igl::MESH_BOOLEAN_TYPE_UNION,V3,V3);
}