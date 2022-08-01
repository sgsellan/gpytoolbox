#ifndef RAY_MESH_INTERSECT_AABB
#define RAY_MESH_INTERSECT_AABB

#include <Eigen/Core>

void ray_mesh_intersect_aabb(const Eigen::MatrixXd& sources, const Eigen::MatrixXd& directions,const Eigen::MatrixXd& VT, const Eigen::MatrixXi& FT, Eigen::VectorXd & ts, Eigen::VectorXi & ids, Eigen::MatrixXd & lambdas);

#endif