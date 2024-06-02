#ifndef POINT_CLOUD_TO_MESH_H
#define POINT_CLOUD_TO_MESH_H

#include <Eigen/Core>

enum class PointCloudReconstructionOuterBoundaryType
{
    Dirichlet,
    Neumann
};

template<typename Real, typename Int,
PointCloudReconstructionOuterBoundaryType outerBoundaryType,
unsigned int dim>
void point_cloud_to_mesh(
    const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& cloud_points,
    const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& cloud_normals,
    const Real screening_weight,
    const int depth,
    const bool parallel,
    const bool verbose,
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& V,
    Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>& F);

#endif