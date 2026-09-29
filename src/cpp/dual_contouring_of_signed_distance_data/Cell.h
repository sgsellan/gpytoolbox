#pragma once
#include "Sphere.h"
#include <Eigen/Core>
#include <vector>
#include <functional>
#include <map>
#include <utility>

// A cubic cell has 8 corners numbered as follows
//
//                 7 -------- 6
//                /|         /|
//               / |        / |
//             4 --------- 5  |
//             |  |       |   |
//             |  3 ------|-- 2
//             | /        |  /
//             |/         | /
//             0 --------- 1
//

// Edges:
// Bottom:   (0-1), (1-2), (2-3), (3-0)
// Top:      (4-5), (5-6), (6-7), (7-4)
// Vertical: (0-4), (1-5), (2-6), (3-7)
static const int EDGE_PAIRS[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom face
    {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top face
    {0, 4}, {1, 5}, {2, 6}, {3, 7}    // vertical edges
};

// Return the two face-direction indices adjacent to the given edge index.
// Face indices: 0:+X,1:-X,2:+Y,3:-Y,4:+Z,5:-Z
std::pair<int, int> get_face_normals_from_edge_idx(int edge_idx);

// Info about closest point from the line segment joining a Hermite sample and the cell vertex to
// the surface of a sphere
struct ClosestPointInfo {
    Eigen::Vector3d p;       // The Hermite intersection point (start of segment)
    Eigen::Vector3d q;       // The closest point on the sphere surface
    Eigen::Vector3d c;       // The sphere center
    Eigen::Vector3d fip;     // The face intersection point
    Eigen::Vector3d barycentric_coords; // (alpha - for v, beta - for p, gamma - for fip) 
    int sphere_idx;          // Index in cell.assigned_spheres
};

class Cell {

    using TrueSdfFunc      = std::function<double(const Eigen::RowVector3d&)>;
    using TrueSdfGradFunc  = std::function<Eigen::RowVector3d(const Eigen::RowVector3d&)>;

public:
    // Grid indices (i,j,k)
    const int ix, iy, iz;

    // Positions of the 8 corners of the cell  in some canonical order
    const Eigen::Matrix<double, 8, 3> corners;

    // SDF values at those corners  in some canonical order
    Eigen::Matrix<double, 8, 1> cornerSDF;
    // SDF normals at those corners  in same canonical order
    Eigen::Matrix<double, 8, 3> cornerNormals;

    // Hermite samples (edge intersection points) in some canonical order
    // Keys are the edge indices (0-11) defined by the global EDGE_PAIRS array (see Cell.cpp)
    std::map<int, Eigen::Vector3d> hermite_positions;
    std::map<int, Eigen::Vector3d> hermite_normals;
    std::map<int, std::vector<Sphere>> hermite_spheres;      // Spheres associated with Hermite samples

    // Assigned spheres
    std::vector<Sphere> assigned_spheres;
    // Closest face indices for assigned spheres
    std::vector<int> closest_faces;

    std::vector<ClosestPointInfo> closest_points_info;

    // Stores pairs of {Face Index (0-5), Intersection Position}
    // Face indices: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
    std::map<int, Eigen::Vector3d> face_intersections;

    std::vector<Eigen::Vector3d> local_mesh_vertices;
    std::vector<Eigen::Vector3i> local_mesh_faces;

    // Cell vertex after QEF minimization
    Eigen::Vector3d vertex = Eigen::Vector3d::Zero();
    double energy;
    Eigen::Vector3d prev_vertex = Eigen::Vector3d::Zero();
    Eigen::Vector3d prev_outer_vertex = Eigen::Vector3d::Zero();
    
    // Does this cell generate a vertex?
    bool has_vertex = false;

    // Current number of inner iterations already performed on this cell. Initialized to 0.
    int inner_iter = 0;

    // this constructor initializes ix, iy, iz, corners, cornerSDF, and empty hermite data, and also has_vertex if there are sign changes
    Cell(int i, int j, int k,
        Eigen::Matrix<double, 8, 3> corner_positions,
        Eigen::Matrix<double, 8, 1> corner_sdf_values,
        Eigen::Matrix<double, 8, 3> corner_sdf_normals);

    // Fill hermite_positions/normals using finite differences and linear interpolation
    void fill_hermite_data(
    const TrueSdfFunc* true_sdf = nullptr,
    const TrueSdfGradFunc* true_sdf_grad = nullptr);


    // compute the centroid of the Hermite data (e.g. for defaults)
    Eigen::Vector3d getCentroid();

    // Solve QEF → sets `vertex`
    void update(const double regularization_weight = 1e-4);

    void minimize_qef(double mu = 0.01, double dc_weight = 0.02,
        double sphere_weight = 1.0, double svd_threshold = 0.01, bool verbose = false);

    Eigen::Vector3d solve_quadratic_system(const Eigen::MatrixXd& A, 
        const Eigen::VectorXd& b, double svd_threshold = 0.01);

    void clean();
};

