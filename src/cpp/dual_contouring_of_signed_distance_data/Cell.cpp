#include "Cell.h"
#include <iostream>
#include <Eigen/Dense>
#include <array>



std::pair<int, int> get_face_normals_from_edge_idx(int edge_idx){
    // 0:+X,1:-X,2:+Y,3:-Y,4:+Z,5:-Z

    // Mapping for the 12 edges -> pair of adjacent face normals
    static const std::array<std::pair<int,int>, 12> edge_to_face_normals = {{
        std::make_pair(3, 5), // edge 0: (0,1)
        std::make_pair(0, 5), // edge 1: (1,2)
        std::make_pair(2, 5), // edge 2: (2,3)
        std::make_pair(1, 5), // edge 3: (3,0)
        std::make_pair(3, 4), // edge 4: (4,5)
        std::make_pair(0, 4), // edge 5: (5,6)
        std::make_pair(2, 4), // edge 6: (6,7)
        std::make_pair(1, 4), // edge 7: (7,4)
        std::make_pair(1, 3), // edge 8: (0,4) vertical
        std::make_pair(0, 3), // edge 9: (1,5) vertical
        std::make_pair(0, 2), // edge 10: (2,6) vertical
        std::make_pair(1, 2)  // edge 11: (3,7) vertical
    }};
    return edge_to_face_normals[edge_idx];
}


// Linear interpolation factor t such that f(a + t(b-a)) = iso
inline double lerp_t(double fa, double fb, double iso = 0.0)
{
    return (fa - iso) / (fa - fb + 1e-16); // avoid division by zero
}

// ============================================================================
// Constructor
// ============================================================================
Cell::Cell(int i, int j, int k,
           Eigen::Matrix<double, 8, 3> corner_positions,
           Eigen::Matrix<double, 8, 1> corner_sdf_values,
           Eigen::Matrix<double, 8, 3> corner_sdf_normals)
    : ix(i), iy(j), iz(k),
      corners(corner_positions),
      cornerSDF(corner_sdf_values),
      cornerNormals(corner_sdf_normals)
{
    // Detect if this cell has a sign change (surface crosses this cell).
    // A zero-crossing exists if some corner has SDF < 0 and another has SDF > 0.
    bool has_neg = false;
    bool has_pos = false;

    for (int c = 0; c < 8; ++c) {
        if (cornerSDF(c) < 0) has_neg = true;
        if (cornerSDF(c) > 0) has_pos = true;
    }

    has_vertex = (has_neg && has_pos);
}


Eigen::Vector3d gradientAtPointUsingTrilinear(
    const Eigen::Matrix<double, 8, 1>& cornerSDF,
    const Eigen::Matrix<double, 8, 3>& corners,
    const Eigen::Vector3d& p)
{
    // Corner SDFs in canonical order:
    // 0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0),
    // 4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)

    const double S000 = cornerSDF(0);
    const double S100 = cornerSDF(1);
    const double S110 = cornerSDF(2);
    const double S010 = cornerSDF(3);
    const double S001 = cornerSDF(4);
    const double S101 = cornerSDF(5);
    const double S111 = cornerSDF(6);
    const double S011 = cornerSDF(7);

    // Recover physical spacing from corners (axis-aligned assumption)
    const double x0 = corners(0, 0);
    const double x1 = corners(1, 0);
    const double y0 = corners(0, 1);
    const double y1 = corners(3, 1);
    const double z0 = corners(0, 2);
    const double z1 = corners(4, 2);

    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double dz = z1 - z0;

    const double eps = 1e-12;

    // Local parametric coords (u,v,w) in [0,1]^3
    double u = 0.5, v = 0.5, w = 0.5; // safe defaults if degenerate
    if (std::abs(dx) > eps) u = (p.x() - x0) / dx;
    if (std::abs(dy) > eps) v = (p.y() - y0) / dy;
    if (std::abs(dz) > eps) w = (p.z() - z0) / dz;

    // Clamp to [0,1] for numerical robustness
    auto clamp01 = [](double t) { return std::max(0.0, std::min(1.0, t)); };
    u = clamp01(u);
    v = clamp01(v);
    w = clamp01(w);

    const double omu = 1.0 - u;
    const double omv = 1.0 - v;
    const double omw = 1.0 - w;

    // Trilinear df/du, df/dv, df/dw in param space
    double df_du =
          (S100 - S000) * omv * omw
        + (S110 - S010) * v   * omw
        + (S101 - S001) * omv * w
        + (S111 - S011) * v   * w;

    double df_dv =
          (S010 - S000) * omu * omw
        + (S110 - S100) * u   * omw
        + (S011 - S001) * omu * w
        + (S111 - S101) * u   * w;

    double df_dw =
          (S001 - S000) * omu * omv
        + (S101 - S100) * u   * omv
        + (S011 - S010) * omu * v
        + (S111 - S110) * u   * v;

    Eigen::Vector3d g = Eigen::Vector3d::Zero();

    // Chain rule: x = x0 + u*dx, etc → df/dx = df/du * du/dx = df/du / dx
    if (std::abs(dx) > eps) g.x() = df_du / dx;
    if (std::abs(dy) > eps) g.y() = df_dv / dy;
    if (std::abs(dz) > eps) g.z() = df_dw / dz;

    return g;
}


// ============================================================================
// Fill Hermite data by checking every edge for a sign change and linearly
// interpolating both intersection position and normals.
// ============================================================================
void Cell::fill_hermite_data(
    const TrueSdfFunc* true_sdf,
    const TrueSdfGradFunc* true_sdf_grad)
{
    hermite_positions.clear();
    hermite_normals.clear();

    if (!has_vertex) {
        return;
    }

    const bool use_true = (true_sdf != nullptr && true_sdf_grad != nullptr);

    // Iterate over all 12 edges
    for (int i = 0; i < 12; ++i) {

        int a = EDGE_PAIRS[i][0];
        int b = EDGE_PAIRS[i][1];

        double fa = cornerSDF(a);
        double fb = cornerSDF(b);

        // No sign change → skip this edge
        if (fa * fb > 0.0) continue;

        Eigen::Vector3d pa = corners.row(a).transpose();
        Eigen::Vector3d pb = corners.row(b).transpose();

        const int edge_index = i; 

        if (!use_true) {
            // ============================
            // 
            // ============================
            double t = lerp_t(fa, fb);  // between 0 and 1

            Eigen::Vector3d p =
                (1.0 - t) * pa + t * pb;

            // Eigen::Vector3d n =
            //     (1.0 - t) * cornerNormals.row(a).transpose() +
            //        t      * cornerNormals.row(b).transpose();

            Eigen::Vector3d n = gradientAtPointUsingTrilinear(cornerSDF, corners, p);

            if (n.norm() > 0.0) {
                n.normalize();
            }

            hermite_positions[edge_index] = p;
            hermite_normals[edge_index] = n;
        } else {
            // ============================
            // TRUE SDF / TRUE GRAD PATH
            // ============================
            Eigen::Vector3d p0 = pa;
            Eigen::Vector3d p1 = pb;

            double f0 = (*true_sdf)(p0);
            double f1 = (*true_sdf)(p1);

            // If even the true SDF doesn’t see a sign change, fall back
            if (f0 * f1 > 0.0) {
            // if (true) {
                double t = lerp_t(fa, fb);

                Eigen::Vector3d p =
                    (1.0 - t) * pa + t * pb;

                Eigen::Vector3d n = gradientAtPointUsingTrilinear(cornerSDF, corners, p);

                if (n.norm() > 0.0) n.normalize();

                hermite_positions[edge_index] = p;
                hermite_normals[edge_index] = n;
                continue;
            }

            // Binary search along the edge for the zero crossing of true_sdf
            const int maxIter = 16;
            const double tol = 1e-6;

            for (int iter = 0; iter < maxIter; ++iter) {
                Eigen::Vector3d pm = 0.5 * (p0 + p1);
                double fm = (*true_sdf)(pm);

                // Stop if small enough
                if (std::abs(fm) < tol) {
                    p0 = p1 = pm;
                    break;
                }

                // Maintain bracket [p0, p1] with sign change
                if (f0 * fm <= 0.0) {
                    p1 = pm;
                    f1 = fm;
                } else {
                    p0 = pm;
                    f0 = fm;
                }

                if ((p1 - p0).norm() < tol) {
                    break;
                }
            }

            Eigen::Vector3d p = 0.5 * (p0 + p1);
            Eigen::Vector3d n = (*true_sdf_grad)(p);

            if (n.norm() > 0.0) {
                n.normalize();
            }

            hermite_positions[edge_index] = p;
            hermite_normals[edge_index] = n;
        }
    }
}

// ============================================================
// update(): QEF minimization placeholder
// ============================================================
void Cell::update(const double regularization_weight)
{
    if (!has_vertex || hermite_positions.empty()) {
        vertex = Eigen::Vector3d::Zero();
        has_vertex = false;
        return;
    }

    const int m = static_cast<int>(hermite_positions.size());
    Eigen::MatrixXd A(m, 3);
    Eigen::VectorXd b(m);

    // Build A and b
    int i = 0;
    for (const auto& pair : hermite_positions) {
        int edge_index = pair.first;
        const Eigen::Vector3d& p = pair.second;
        
        // Get normal using the edge_index key from the hermite_normals map
        Eigen::Vector3d n = hermite_normals.at(edge_index); 

        if (n.norm() > 0) {
            n.normalize(); // Normalize the copy
        }

        A.row(i) = n.transpose();
        b(i)     = n.dot(p);
        i++;
    }

    vertex = solve_quadratic_system(A, b, 0.01);

    // debug overload
    // vertex = centroid;

    has_vertex = true;
}



Eigen::Vector3d Cell::solve_quadratic_system(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double svd_threshold
) {
    Eigen::Matrix3d ATA = A.transpose() * A;
    Eigen::Vector3d ATb = A.transpose() * b;

    // Compute SVD of AtA
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector3d S = svd.singularValues();
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Vector3d S_inv;
    for (int i = 0; i < 3; ++i) {
        if (S(i) > svd_threshold)
            S_inv(i) = 1.0 / S(i);
        else
            S_inv(i) = 0.0;                // truncate small singular values
    }

    //std::cout << "Singular values: " << S_inv.transpose() << std::endl;
    // Pseudoinverse of ATA:
    // (ATA)^+ = V * diag(S_inv) * U^T
    Eigen::Matrix3d ATA_pinv = V * S_inv.asDiagonal() * U.transpose();
    Eigen::Vector3d centroid = getCentroid();

    // “Right-hand side” centered at centroid:
    // c = (ATA)^+ (ATb - ATA * centroid)
    Eigen::Vector3d rhs = ATb - ATA * centroid;
    Eigen::Vector3d c   = ATA_pinv * rhs;

    // Final solution:
    return centroid + c;
}


void Cell::minimize_qef(
    double mu,
    double dc_weight,
    double sphere_weight,
    double svd_threshold,
    bool verbose
) {
    // Build positions and normals in a consistent order by iterating over
    // hermite_positions and looking up the corresponding normal by the same key.
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3d> normals;
    positions.reserve(hermite_positions.size());
    normals.reserve(hermite_positions.size());

    for (const auto& kv : hermite_positions) {
        int edge_index = kv.first;
        positions.push_back(kv.second);

        // Find matching normal. If missing, push a zero vector as a fallback.
        auto nit = hermite_normals.find(edge_index);
        if (nit != hermite_normals.end()) {
            Eigen::Vector3d n = nit->second;
            normals.push_back(n);
        } else {
            normals.push_back(Eigen::Vector3d::Zero());
        }
    }

    int num_normals = (int) normals.size();
    int num_spheres = (int) closest_points_info.size();
    int num_eqs_regularization = 3;
    Eigen::MatrixXd A(num_normals + num_spheres + num_eqs_regularization, 3);
    Eigen::VectorXd b(num_normals + num_spheres + num_eqs_regularization);

    // Initialize A and b to zero
    A.setZero();
    b.setZero();

    int current_row = 0;
    for (size_t i = 0; i < num_normals; ++i) {
        Eigen::Vector3d scaled_normal = dc_weight * normals[i];
        A.row(current_row) = scaled_normal.transpose();
        b(current_row) = scaled_normal.dot(positions[i]);
        current_row++;
    }

    const double sqrt_sphere_weight = std::sqrt(sphere_weight);
    //std::cout << "Number of sphere constraints: " << num_spheres << std::endl;
    for (const auto& info : closest_points_info) {
        double alpha = info.barycentric_coords(0);
        double beta  = info.barycentric_coords(1);
        double gamma = info.barycentric_coords(2);

        if (std::abs(alpha) < 1e-6){
            // alpha = 0.001;
            alpha = 1.0;
            beta = 0.0;
            gamma = 0.0;
        }

        Eigen::Vector3d t = alpha * prev_vertex + beta * info.p + gamma * info.fip;
        Eigen::Vector3d c_to_t_mesh = t - info.c;
        double rho = c_to_t_mesh.norm();

        // radius is the distance between q and c
        double radius = (info.q - info.c).norm();

        if (rho > radius) {
            // Get the index of the sphere in cell.assigned_spheres
            int sphere_idx = info.sphere_idx;
            // Check the sign of the sphere
            int sign = assigned_spheres[sphere_idx].sign;
            if (sign == -1){
                continue;
            }
        }

        // Calculate q: the point on the sphere surface closest to t_mesh
        Eigen::Vector3d q_sphere;
        if (rho < 1e-9) {
            q_sphere = info.c; 
        } else {
            q_sphere = info.c + (c_to_t_mesh / rho) * radius;
        }
        
        Eigen::Vector3d q = q_sphere;
        Eigen::Vector3d d = q - info.c;
        Eigen::Vector3d p = info.p;
        Eigen::Vector3d fip = info.fip;

        A.row(current_row) = alpha * sqrt_sphere_weight * d.transpose();
        double b_row_not_scaled = q.dot(d) 
                                 - beta * p.dot(d)
                                 - gamma * fip.dot(d);
        b(current_row) = sqrt_sphere_weight * b_row_not_scaled;
        current_row++;
    }

    double sqrt_mu = std::sqrt(mu);
    for (int i = 0; i < 3; ++i) {
        A.row(current_row) = sqrt_mu * Eigen::Vector3d::Unit(i).transpose();
        b(current_row) = sqrt_mu * prev_vertex(i);
        current_row++;
    }   

    vertex = solve_quadratic_system(A, b, svd_threshold);

    Eigen::VectorXd residual = A * vertex - b;
    energy = residual.squaredNorm();
}


// void Cell::minimize_qef(
//     double mu,
//     double dc_weight,
//     double sphere_weight,
//     double svd_threshold,
//     bool verbose
// ) {
//     // Build positions and normals in a consistent order by iterating over
//     // hermite_positions and looking up the corresponding normal by the same key.
//     std::vector<Eigen::Vector3d> positions;
//     std::vector<Eigen::Vector3d> normals;
//     positions.reserve(hermite_positions.size());
//     normals.reserve(hermite_positions.size());

//     for (const auto& kv : hermite_positions) {
//         int edge_index = kv.first;
//         positions.push_back(kv.second);

//         // Find matching normal. If missing, push a zero vector as a fallback.
//         auto nit = hermite_normals.find(edge_index);
//         if (nit != hermite_normals.end()) {
//             Eigen::Vector3d n = nit->second;
//             normals.push_back(n);
//         } else {
//             normals.push_back(Eigen::Vector3d::Zero());
//         }
//     }

//     int num_normals = (int) normals.size();
//     int num_spheres = (int) closest_points_info.size();
//     int num_eqs_regularization = 3;
//     Eigen::MatrixXd A(num_spheres * 3, 3);
//     Eigen::VectorXd b(num_spheres * 3);

//     int current_row = 0;
//     //std::cout << "Number of normals: " << num_normals << std::endl;
//     // for (size_t i = 0; i < num_normals; ++i) {
//     //     Eigen::Vector3d scaled_normal = dc_weight * normals[i];
//     //     A.row(current_row) = scaled_normal.transpose();
//     //     b(current_row) = scaled_normal.dot(positions[i]);
//     //     current_row++;
//     // }

//     const double sqrt_sphere_weight = std::sqrt(sphere_weight);
//     //std::cout << "Number of sphere constraints: " << num_spheres << std::endl;
//     for (const auto& info : closest_points_info) {
//         double alpha = info.barycentric_coords(0);
//         double beta  = info.barycentric_coords(1);
//         double gamma = info.barycentric_coords(2);

//         if (abs(alpha) < 1e-6){
//             // alpha = 0.001;
//             alpha = 1.0;
//             beta = 0.0;
//             gamma = 0.0;
//         }

//         Eigen::Vector3d t = alpha * prev_vertex + beta * info.p + gamma * info.fip;
//         Eigen::Vector3d c_to_t_mesh = t - info.c;
//         double rho = c_to_t_mesh.norm();

//         // Calculate q: the point on the sphere surface closest to t_mesh
//         Eigen::Vector3d q_sphere;
//         if (rho < 1e-9) {
//             q_sphere = info.c; 
//         } else {
//             // radius is the distance between q and c
//             double radius = (info.q - info.c).norm();
//             q_sphere = info.c + (c_to_t_mesh / rho) * radius;
//         }

        
//         Eigen::Vector3d q = q_sphere;
//         Eigen::Vector3d d = q - info.c;
//         Eigen::Vector3d p = info.p;
//         Eigen::Vector3d fip = info.fip;

//         // A.row(current_row) = alpha * sqrt_sphere_weight * d.transpose();
//         // double b_row_not_scaled = q.dot(d) 
//         //                          - beta * p.dot(d)
//         //                          - gamma * fip.dot(d);
//         // b(current_row) = sqrt_sphere_weight * b_row_not_scaled;
//         // current_row++;

//         A.block<3,3>(current_row, 0) =
//             alpha * sqrt_sphere_weight * Eigen::Matrix3d::Identity();

//         b.segment<3>(current_row) =
//             sqrt_sphere_weight * (q - beta * p - gamma * fip);
//         current_row += 3;
//     }

//     // double sqrt_mu = std::sqrt(mu);
//     // for (int i = 0; i < 3; ++i) {
//     //     A.row(current_row) = sqrt_mu * Eigen::Vector3d::Unit(i).transpose();
//     //     b(current_row) = sqrt_mu * prev_vertex(i);
//     //     current_row++;
//     // }   

//     //Print A and b
//     // std::cout << "Matrix A\n" << A << std::endl << std::endl;
//     // std::cout << "Vector b" << b << std::endl  << std::endl << std::endl;

//     vertex = solve_quadratic_system(A, b, svd_threshold);

//     Eigen::VectorXd residual = A * vertex - b;
//     energy = residual.squaredNorm();
// }


Eigen::Vector3d Cell::getCentroid(){
    // centroid of hermite positions
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    if(hermite_positions.empty()){
        return centroid;
    }
    
    // Iterate over the values (Vector3d) stored in the map
    for(const auto& pair : hermite_positions){
        centroid += pair.second;
    }
    
    centroid /= double(hermite_positions.size());
    return centroid;
}

void Cell::clean() {
    closest_points_info.clear();
    assigned_spheres.clear();
    closest_faces.clear();
    local_mesh_vertices.clear();
    local_mesh_faces.clear();
    inner_iter = 0;
}