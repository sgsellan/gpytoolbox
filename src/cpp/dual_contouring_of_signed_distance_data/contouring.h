#pragma once

#include <Eigen/Core>
#include <functional>

using TrueSdfFunc      = std::function<double(const Eigen::RowVector3d&)>;
using TrueSdfGradFunc  = std::function<Eigen::RowVector3d(const Eigen::RowVector3d&)>;

// Function type used to triangulate a face matrix F when needed.
// Takes the original face matrix and the vertex matrix V and returns a triangle-only face matrix.
using TriangulateFunc = std::function<Eigen::MatrixXi(const Eigen::MatrixXi&, const Eigen::MatrixXd&)>;

// Public triangulation helpers (choose one when calling assign_spheres_to_cells)
Eigen::MatrixXi triangulate_v1(const Eigen::MatrixXi &F, const Eigen::MatrixXd &V);
Eigen::MatrixXi triangulate_v2(const Eigen::MatrixXi &F, const Eigen::MatrixXd &V);


// Options struct — can be extended later
struct ContouringOptions {
    bool verbose = false;
    double mu = 0.1;
    double dc_weight = 0.02;
    double sphere_weight = 1.0;
    double svd_threshold = 0.01;
    int outer_iters = 100;
    int inner_iters = 100;
    bool hermite_update = true;
    double new_hermite_pos_weight = 0.2;
    double new_face_pos_weight = 0.2;
    double new_hermite_normal_weight = 0.2;
    int batch_size = 200000;
};

// A single, simple function that mimics the libigl marching_cubes API
void contouring(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    double isoValue,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    const ContouringOptions& options,
    const TrueSdfFunc& true_sdf,
    const TrueSdfGradFunc& true_sdf_grad
);

Eigen::Vector3d gradientAt(
    const Eigen::MatrixXd &GV, 
    const Eigen::VectorXd &S,   
    int i, int j, int k,
    int resX, int resY, int resZ
);

void assign_spheres_to_cells(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::vector<class Cell>& cells,
    int resX,
    int resY,
    int resZ,
    int batch_size,
    Eigen::MatrixXi& TriF
);

void compute_face_cell_intersections(
    std::vector<class Cell>& cells,
    int resX,
    int resY,
    int resZ,
    double weight_new_pos
);

// Extract mesh vertices & quad faces from cells. GV is the grid vertex positions used for
// gradient/normals and spatial computations.
void extract_mesh_from_cells(
    const std::vector<class Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    std::vector<Eigen::Vector3d>& hermite_normals
);

void extract_mesh_from_cells(
    const std::vector<class Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F
);

// Variant that also returns a per-vertex mapping to the originating cell index
void extract_mesh_from_cells(
    const std::vector<class Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    std::vector<int>* vertexCellIndex
);

void start_outer_iteration(
    std::vector<class Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int resX,
    int resY,
    int resZ
);

void update_hermite_points(
    std::vector<Cell>& cells,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& TriF,
    double weight_new_pos
);

// overload without true_sdf and true_sdf_grad
void contouring(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    double isoValue,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    const ContouringOptions& options
);

void show_total_energy(const std::vector<Cell>& cells, bool verbose);


void optimize_triangulation(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXi& TriF,
    int resX,
    int resY,
    int resZ
);


void orient_triangles_for_quad(
    Eigen::MatrixXi &TriF, 
    int tri_row0, 
    int tri_row1,
    int a, int b, int c, int d,
    const Eigen::MatrixXd &V
);


double compute_total_distance_to_spheres(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& TriF,
    int resX,
    int resY,
    int resZ
);


double cell_diagonal(const Eigen::MatrixXd& GV, int resX, int resY, int resZ,
    double& min_x, double& min_y, double& min_z,
    double& dx, double& dy, double& dz);