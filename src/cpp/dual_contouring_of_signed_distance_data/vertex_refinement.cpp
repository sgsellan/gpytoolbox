#include "vertex_refinement.h" 
#include "Cell.h" 
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <map>
#include <cmath>
#include <igl/point_mesh_squared_distance.h>
#include <igl/AABB.h>
#include <stdexcept>


// Indices: [0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]
static const int FACE_CORNERS[6][4] = {
    {1, 5, 6, 2}, 
    {0, 3, 7, 4}, 
    {2, 6, 7, 3},
    {0, 4, 5, 1}, 
    {4, 7, 6, 5}, 
    {0, 1, 2, 3}
};

static int get_edge_index(int c1, int c2) {
    for (int i = 0; i < 12; ++i) {
        // EDGE_PAIRS is defined in Cell.h
        int p1 = EDGE_PAIRS[i][0];
        int p2 = EDGE_PAIRS[i][1];
        // Check both (p1 -> p2) and (p2 -> p1) for symmetry
        if ((c1 == p1 && c2 == p2) || (c1 == p2 && c2 == p1)) {
            return i;
        }
    }
    return -1; 
}


Eigen::Vector3d compute_barycentric_coords(
    const Eigen::Vector3d& P, 
    const Eigen::Vector3d& V0, 
    const Eigen::Vector3d& V1, 
    const Eigen::Vector3d& V2
) {
    // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates

    Eigen::Vector3d V0V1 = V1 - V0;
    Eigen::Vector3d V0V2 = V2 - V0;
    Eigen::Vector3d V0P = P - V0;

    double d11 = V0V1.dot(V0V1);
    double d12 = V0V1.dot(V0V2);
    double d22 = V0V2.dot(V0V2);
    double dP1 = V0P.dot(V0V1);
    double dP2 = V0P.dot(V0V2);
    
    double denom = d11 * d22 - d12 * d12;
    
    if (std::abs(denom) < 1e-12) {
        return Eigen::Vector3d(1.0, 0.0, 0.0); 
    }

    double beta = (d22 * dP1 - d12 * dP2) / denom;
    double gamma = (d11 * dP2 - d12 * dP1) / denom;

    double alpha = 1.0 - beta - gamma;

    return Eigen::Vector3d(alpha, beta, gamma);
}



// New batched API implementation
void closest_points_on_mesh(
    const Eigen::MatrixXd& V_mesh,
    const Eigen::MatrixXi& F_mesh,
    const Eigen::MatrixXd& targets,               // Nx3
    Eigen::MatrixXd& closest_points_out,          // Nx3
    Eigen::VectorXi& face_indices_out             // N
) {
    // Initialize outputs
    closest_points_out = targets; // Default to input
    face_indices_out.resize(targets.rows());
    face_indices_out.setConstant(-1);

    if (F_mesh.rows() == 0 || V_mesh.rows() == 0 || targets.rows() == 0) return;

    Eigen::VectorXd sq_distances;
    Eigen::VectorXi I;            // Index of the closest face per query
    Eigen::MatrixXd C;            // Closest point coordinates (rows = queries)

    // igl supports multiple query points at once
    igl::point_mesh_squared_distance(targets, V_mesh, F_mesh, sq_distances, I, C);

    if (C.rows() == targets.rows()) {
        closest_points_out = C;
    } else if (C.rows() > 0) {
        throw std::runtime_error("Mismatch in number of closest points returned.");
    }

    if (I.size() == targets.rows()) {
        face_indices_out = I;
    } else if (I.size() > 0) {
        throw std::runtime_error("Mismatch in number of face indices returned.");
    }
}


static std::vector<ClosestPointInfo> compute_sphere_to_mesh_closest_points(
    const Cell& cell,
    const Eigen::MatrixXd& V_mesh,
    const Eigen::MatrixXi& F_mesh
) {
    std::vector<ClosestPointInfo> results;

    // Check if mesh is valid before running query
    if (V_mesh.rows() == 0 || F_mesh.rows() == 0) return results;

    const size_t N = cell.assigned_spheres.size();
    if (N == 0) return results;

    // Batch all sphere centers into a matrix
    Eigen::MatrixXd centers(N, 3);
    for (size_t i = 0; i < N; ++i) centers.row(i) = cell.assigned_spheres[i].center.transpose();

    // Call batched closest-point query once
    Eigen::MatrixXd closest_pts;
    Eigen::VectorXi face_indices;
    closest_pts.resize(N, 3);
    face_indices.resize(N);
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(V_mesh, F_mesh);
    for (size_t i = 0; i < N; ++i) {
        Eigen::RowVector3d c = centers.row(i);
        int fid;
        Eigen::RowVector3d cp;
        tree.squared_distance(V_mesh, F_mesh, c, fid, cp);
        closest_pts.row(i) = cp;
        face_indices(i) = fid;
    }
    
    // closest_points_on_mesh(V_mesh, F_mesh, centers, closest_pts, face_indices);

    // Process each sphere using returned closest points
    for (size_t i = 0; i < N; ++i) {
        const auto& sphere = cell.assigned_spheres[i];
        Eigen::Vector3d c = sphere.center;
        double radius = sphere.radius;

        //Eigen::Vector3d t_mesh = (closest_pts.rows() > (int)i) ? closest_pts.row(i).transpose() : c;
        Eigen::Vector3d t_mesh = closest_pts.row(i).transpose();
        //int tri_idx = (face_indices.size() > (int)i) ? face_indices(i) : -1;
        int tri_idx = face_indices(i);

        // Calculate the vector from sphere center (c) to the closest mesh point (t_mesh)
        Eigen::Vector3d c_to_t_mesh = t_mesh - c;
        double rho = c_to_t_mesh.norm();

        // Calculate q: the point on the sphere surface closest to t_mesh
        Eigen::Vector3d q_sphere;
        if (rho < 1e-9) {
            q_sphere = c; 
        } else {
            q_sphere = c + (c_to_t_mesh / rho) * radius;
        }

        // Calculate the barycentric coordinates of t_mesh with respect to the triangle where it lies
        Eigen::Vector3d b_coords = Eigen::Vector3d::Zero();
        Eigen::Vector3d fip_coord = Eigen::Vector3d::Zero();
        
        if (tri_idx < 0) {   // Could happen if query failed
            std::cerr << "Warning: Could not find closest triangle for sphere index " << i << std::endl;
            continue;
        }

        // Vertex indices in order [v, hermite point, face intersection point]
        const Eigen::Vector3i& tri_indices = F_mesh.row(tri_idx);
        
        Eigen::Vector3d V0 = V_mesh.row(tri_indices(0)).transpose();
        Eigen::Vector3d V1 = V_mesh.row(tri_indices(1)).transpose();
        fip_coord = V_mesh.row(tri_indices(2)).transpose();
        
        b_coords = compute_barycentric_coords(t_mesh, V0, V1, fip_coord);
    
        ClosestPointInfo info;
        info.p = V1;           
        info.q = q_sphere;         
        info.c = c;
        info.fip = fip_coord;
        info.barycentric_coords = b_coords;
        info.sphere_idx = static_cast<int>(i);

        results.push_back(info);
    }

    return results;
}


void refine_vertex_from_face_intersections(
    Cell& cell
){
    if (!cell.has_vertex || cell.face_intersections.empty()) return;

    // -----------------------------------------------------
    // Contruct local triangle mesh 
    // -----------------------------------------------------

    std::vector<Eigen::Vector3d> local_vertices;
    std::vector<Eigen::Vector3i> local_faces;

    // Add cell vertex as first local vertex
    local_vertices.push_back(cell.vertex);
    const int v_idx = 0;     

    // Map to store the local vertex index (1, 2, 3...) for each global Hermite edge index (0-11)
    std::map<int, int> hermite_edge_to_local;
    int current_local_idx = 1;
    
    // Add Hermite Points (H) to local list, using edge index as the key.
    for (const auto& hermite_data : cell.hermite_positions) {
        const int edge_idx = hermite_data.first;
        const Eigen::Vector3d& position = hermite_data.second;

        local_vertices.push_back(position);
        hermite_edge_to_local[edge_idx] = current_local_idx++;
    }

    // Triangulate face intersections (F)
    for (const auto& face_data : cell.face_intersections) {
        const int face_idx = face_data.first;
        const Eigen::Vector3d& intersection_point = face_data.second;

        // Add intersection point to local vertices
        local_vertices.push_back(intersection_point);
        const int F_idx = current_local_idx++;    // Index of intersection point in local vertices


        const int c1 = FACE_CORNERS[face_idx][0];
        const int c2 = FACE_CORNERS[face_idx][1];
        const int c3 = FACE_CORNERS[face_idx][2];
        const int c4 = FACE_CORNERS[face_idx][3];
        
        const int corner_pairs[4][2] = {{c1, c2}, {c2, c3}, {c3, c4}, {c4, c1}};

        for(int edge_i = 0; edge_i < 4; ++edge_i) {
            // Find the global edge index from the corner pair
            int edge_idx = get_edge_index(corner_pairs[edge_i][0], corner_pairs[edge_i][1]);
            
            // Check if a Hermite point exists on this edge using the map
            auto it = hermite_edge_to_local.find(edge_idx);
            
            if (it != hermite_edge_to_local.end()) {
                const int H_idx = it->second; 
                local_faces.push_back(Eigen::Vector3i(v_idx, H_idx, F_idx));
            }
        }
    }

    if (local_faces.empty()) return;

    // Assign local mesh data to cell
    cell.local_mesh_vertices = local_vertices;
    cell.local_mesh_faces = local_faces;


    // Convert local vertices and faces to Eigen matrices for processing with libgl
    Eigen::MatrixXd V_mesh(local_vertices.size(), 3);
    for (size_t l = 0; l < local_vertices.size(); ++l) {
        V_mesh.row(l) = local_vertices[l].transpose();
    }

    Eigen::MatrixXi F_mesh(local_faces.size(), 3);
    for (size_t l = 0; l < local_faces.size(); ++l) {
        F_mesh.row(l) = local_faces[l];
    }

    cell.closest_points_info = compute_sphere_to_mesh_closest_points(cell, V_mesh, F_mesh);
}

