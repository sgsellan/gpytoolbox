#pragma once

#include "Cell.h"
#include <vector>


void refine_vertex_from_face_intersections(
    Cell& cell
);

void closest_points_on_mesh(
    const Eigen::MatrixXd& V_mesh,
    const Eigen::MatrixXi& F_mesh,
    const Eigen::MatrixXd& targets,               // Nx3 query points
    Eigen::MatrixXd& closest_points_out,          // Nx3 closest points
    Eigen::VectorXi& face_indices_out             // N face indices
);

void compute_barycentric_coords(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c,
    Eigen::Vector3d& barycentric_coords_out
);

Eigen::Vector3d compute_barycentric_coords(
    const Eigen::Vector3d& P, 
    const Eigen::Vector3d& V0, 
    const Eigen::Vector3d& V1, 
    const Eigen::Vector3d& V2
);

// Single-target wrapper
// void closest_point_on_mesh(
//     const Eigen::MatrixXd& V_mesh, 
//     const Eigen::MatrixXi& F_mesh, 
//     const Eigen::Vector3d& target,
//     Eigen::Vector3d& closest_point,   
//     int& face_index
// );