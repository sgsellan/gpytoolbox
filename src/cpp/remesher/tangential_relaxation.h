#ifndef TANGENTIAL_RELAXATION
#define TANGENTIAL_RELAXATION



#include <Eigen/Core>

// Tangential smoothing. Fixed `feature` vertices and feature-edge junction /
// corner vertices (feature degree != 2) are not moved. Feature-line vertices
// (feature degree == 2) are smoothed along the feature and reprojected onto the
// original feature polyline (`featV0`, `featE0`) when `project` is true. All
// other vertices are smoothed on their tangent plane and (when `project`)
// reprojected onto the surface (`V0`, `F0`).
void tangential_relaxation(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges,
Eigen::MatrixXd & V0 ,Eigen::MatrixXi & F0, Eigen::VectorXd & lambda,
Eigen::MatrixXd & featV0, Eigen::MatrixXi & featE0, bool project);


#endif
