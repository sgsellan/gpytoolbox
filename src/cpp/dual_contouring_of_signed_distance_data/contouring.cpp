#include "contouring.h"
#include "Cell.h"
#include "vertex_refinement.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <igl/point_mesh_squared_distance.h>
#include <igl/AABB.h>
#include <Eigen/Core>
#include <random>
#include <stdexcept>
#include "hermite_update.h"

inline int index3D(int i, int j, int k, int resX, int resY, int resZ) {
    return i + resX * (j + resY * k);
}

inline int cellIndex3D(int i, int j, int k, int resX, int resY, int resZ) {
    // There are (resX-1) x (resY-1) x (resZ-1) cells
    return i + (resX - 1) * (j + (resY - 1) * k);
}


// Orient the two triangles forming a quad so that they have a consistent normal with respect
// to the quad's normal computed from the original quad vertices.
void orient_triangles_for_quad(
    Eigen::MatrixXi &TriF, 
    int tri_row0, 
    int tri_row1,
    int a, int b, int c, int d,
    const Eigen::MatrixXd &V
) {
    if (V.rows() == 0) return;

    Eigen::Vector3d pa = V.row(a).transpose();
    Eigen::Vector3d pb = V.row(b).transpose();
    Eigen::Vector3d pc = V.row(c).transpose();
    Eigen::Vector3d pd = V.row(d).transpose();

    Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
    if (quad_n.squaredNorm() < 1e-24) {
        // Try the other diagonal
        quad_n = (pc - pa).cross(pd - pa);
    }
    if (quad_n.squaredNorm() < 1e-24) return;    // Degenerate quad, we give up

    {
        // Orient tri 0
        Eigen::Vector3d p1 = V.row(TriF(tri_row0, 0)).transpose();
        Eigen::Vector3d p2 = V.row(TriF(tri_row0, 1)).transpose();
        Eigen::Vector3d p3 = V.row(TriF(tri_row0, 2)).transpose();
        Eigen::Vector3d tri_n = (p2 - p1).cross(p3 - p1);
        if (tri_n.dot(quad_n) < 0.0) {
            int i1 = TriF(tri_row0, 1);
            TriF(tri_row0, 1) = TriF(tri_row0, 2);
            TriF(tri_row0, 2) = i1;
        }
    }

    {
        // Orient tri 1
        Eigen::Vector3d p1 = V.row(TriF(tri_row1, 0)).transpose();
        Eigen::Vector3d p2 = V.row(TriF(tri_row1, 1)).transpose();
        Eigen::Vector3d p3 = V.row(TriF(tri_row1, 2)).transpose();
        Eigen::Vector3d tri_n = (p2 - p1).cross(p3 - p1);
        if (tri_n.dot(quad_n) < 0.0) {
            int i1 = TriF(tri_row1, 1);
            TriF(tri_row1, 1) = TriF(tri_row1, 2);
            TriF(tri_row1, 2) = i1;
        }
    }
}

Eigen::MatrixXi triangulate_v1(const Eigen::MatrixXi &F, const Eigen::MatrixXd &V) {
    Eigen::MatrixXi TriF;
    if (F.cols() == 4) {
        TriF.resize(F.rows() * 2, 3);
        for (int f = 0; f < F.rows(); ++f) {
            int a = F(f, 0);
            int b = F(f, 1);
            int c = F(f, 2);
            int d = F(f, 3);
            TriF.row(2 * f)     << a, b, c;
            TriF.row(2 * f + 1) << a, c, d;
        }
        // Orient each quad's triangles consistently using V
        for (int f = 0; f < F.rows(); ++f) {
            int a = F(f, 0);
            int b = F(f, 1);
            int c = F(f, 2);
            int d = F(f, 3);
            orient_triangles_for_quad(TriF, 2 * f, 2 * f + 1, a, b, c, d, V);
        }
    } else {
        // For triangles, edges or points we can just copy.
        TriF = F;
    }
    return TriF;
}


Eigen::MatrixXi triangulate_v2(const Eigen::MatrixXi &F, const Eigen::MatrixXd &V) {
    // Same as v1 but using the other diagonal to split quads.
    Eigen::MatrixXi TriF;
    if (F.cols() == 4) {
        TriF.resize(F.rows() * 2, 3);
        for (int f = 0; f < F.rows(); ++f) {
            int a = F(f, 0);
            int b = F(f, 1);
            int c = F(f, 2);
            int d = F(f, 3);
            // Use consistent winding for the bd diagonal: (b,c,d) and (b,d,a)
            TriF.row(2 * f)     << b, c, d;
            TriF.row(2 * f + 1) << b, d, a;

            // Orient the two triangles consistently with respect to the quad normal
            orient_triangles_for_quad(TriF, 2 * f, 2 * f + 1, a, b, c, d, V);
        }
    } else {
        TriF = F;
    }
    return TriF;
}


// Helper: compute gradient of S at grid index (i,j,k) via finite differences
Eigen::Vector3d gradientAt(
    const Eigen::MatrixXd &GV, 
    const Eigen::VectorXd &S,   
    int i, int j, int k,
    int resX, int resY, int resZ
) {
    auto clamp = [](int v, int lo, int hi) {
        return std::max(lo, std::min(hi, v));
    };

    // Neighbor indices in each direction
    int im = clamp(i - 1, 0, resX - 1);
    int ip = clamp(i + 1, 0, resX - 1);
    int jm = clamp(j - 1, 0, resY - 1);
    int jp = clamp(j + 1, 0, resY - 1);
    int km = clamp(k - 1, 0, resZ - 1);
    int kp = clamp(k + 1, 0, resZ - 1);

    // Index to flat
    auto idx3 = [&](int ii, int jj, int kk) {
        return index3D(ii, jj, kk, resX, resY, resZ);
    };

    // Positions
    Eigen::Vector3d x_im = GV.row(idx3(im, j,  k));
    Eigen::Vector3d x_ip = GV.row(idx3(ip, j,  k));
    Eigen::Vector3d x_jm = GV.row(idx3(i,  jm, k));
    Eigen::Vector3d x_jp = GV.row(idx3(i,  jp, k));
    Eigen::Vector3d x_km = GV.row(idx3(i,  j,  km));
    Eigen::Vector3d x_kp = GV.row(idx3(i,  j,  kp));

    // Values
    double s_im = S(idx3(im, j,  k));
    double s_ip = S(idx3(ip, j,  k));
    double s_jm = S(idx3(i,  jm, k));
    double s_jp = S(idx3(i,  jp, k));
    double s_km = S(idx3(i,  j,  km));
    double s_kp = S(idx3(i,  j,  kp));

    Eigen::Vector3d g = Eigen::Vector3d::Zero();

    // dS/dx
    double dx = (x_ip - x_im).x();
    if (std::abs(dx) > 1e-12) {
        g.x() = (s_ip - s_im) / dx;
    }

    // dS/dy
    double dy = (x_jp - x_jm).y();
    if (std::abs(dy) > 1e-12) {
        g.y() = (s_jp - s_jm) / dy;
    }

    // dS/dz
    double dz = (x_kp - x_km).z();
    if (std::abs(dz) > 1e-12) {
        g.z() = (s_kp - s_km) / dz;
    }

    // Normalize to get a unit normal
    double nrm = g.norm();
    if (nrm > 0.0) {
        g /= nrm;
    }

    return g;
}


std::vector<Cell> generate_cells(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    const TrueSdfFunc& true_sdf,
    const TrueSdfGradFunc& true_sdf_grad)
{
    std::vector<Cell> cells;
    cells.reserve((resX - 1) * (resY - 1) * (resZ - 1));

    for (int k = 0; k < resZ - 1; ++k) {
        for (int j = 0; j < resY - 1; ++j) {
            for (int i = 0; i < resX - 1; ++i) {

                // --- Gather the 8 corners ---
                Eigen::Matrix<double, 8, 3> corner_positions;
                Eigen::Matrix<double, 8, 1> corner_sdf;
                Eigen::Matrix<double, 8, 3> corner_normals;

                // Offsets for the 8 corners relative to (i,j,k)
                const int off[8][3] = {
                    {0, 0, 0},
                    {1, 0, 0},
                    {1, 1, 0},
                    {0, 1, 0},
                    {0, 0, 1},
                    {1, 0, 1},
                    {1, 1, 1},
                    {0, 1, 1}
                };

                for (int c = 0; c < 8; ++c) {
                    int gi = i + off[c][0];
                    int gj = j + off[c][1];
                    int gk = k + off[c][2];

                    int flatIdx = index3D(gi, gj, gk, resX, resY, resZ);

                    corner_positions.row(c) = GV.row(flatIdx);
                    corner_sdf(c) = S(flatIdx);
                    corner_normals.row(c) = gradientAt(GV, S, gi, gj, gk, resX, resY, resZ);
                }

                // --- Construct the Cell ---
                Cell cell(i, j, k, corner_positions, corner_sdf, corner_normals);

                // Fill Hermite data now (uses corner positions + normals)
                if(true_sdf && true_sdf_grad){
                    cell.fill_hermite_data(&true_sdf, &true_sdf_grad);
                } else {
                    cell.fill_hermite_data();
                }

                cells.push_back(std::move(cell));
            }
        }
    }

    return cells;
}

void extract_mesh_from_cells(
    const std::vector<Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F)
{
    auto cellIndex = [&](int i, int j, int k) {
        return cellIndex3D(i, j, k, resX, resY, resZ);
    };

    const int cx = resX - 1;
    const int cy = resY - 1;
    const int cz = resZ - 1;

    // 1) Collect vertices (unchanged)
    std::vector<Eigen::Vector3d> verts;
    verts.reserve(cells.size());
    std::vector<int> cellVertexIndex(cells.size(), -1);

    for (int k = 0; k < cz; ++k) {
        for (int j = 0; j < cy; ++j) {
            for (int i = 0; i < cx; ++i) {
                int ci = cellIndex(i, j, k);
                const Cell& c = cells[ci];
                if (!c.has_vertex) continue;

                cellVertexIndex[ci] = (int)verts.size();
                verts.push_back(c.vertex);
            }
        }
    }

    V.resize((int)verts.size(), 3);
    for (int i = 0; i < (int)verts.size(); ++i) {
        V.row(i) = verts[i].transpose();
    }

    // 2) Helper: check if an edge between two grid vertices has a sign change
    auto edgeHasSignChange = [&](int gv0, int gv1, double iso) {
        double s0 = S(gv0) - iso;
        double s1 = S(gv1) - iso;
        // require opposite signs (and both nonzero)
        return (s0 * s1 < 0.0);
    };

    auto getCellVert = [&](int i, int j, int k) -> int {
        if (i < 0 || i >= cx ||
            j < 0 || j >= cy ||
            k < 0 || k >= cz) return -1;
        int ci = cellIndex(i, j, k);
        return cellVertexIndex[ci];
    };

    std::vector<Eigen::Vector4i> quads;
    quads.reserve(cells.size() * 3);

    // 3) Build quads **only** around sign–changing edges

    auto idxV = [&](int i, int j, int k) {
        return index3D(i, j, k, resX, resY, resZ);
    };

    double iso = 0.0; // or pass isoValue in as a parameter if you want flexibility

    // --- Edges along X ---
    // edge between grid verts (i,j,k) and (i+1,j,k)
    for (int i = 0; i < cx; ++i) {
        for (int j = 1; j < cy; ++j) {
            for (int k = 1; k < cz; ++k) {

                int gv0 = idxV(i,   j, k);
                int gv1 = idxV(i+1, j, k);
                if (!edgeHasSignChange(gv0, gv1, iso)) continue;

                int c0 = getCellVert(i, j-1, k-1);
                int c1 = getCellVert(i, j,   k-1);
                int c2 = getCellVert(i, j,   k  );
                int c3 = getCellVert(i, j-1, k  );

                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
                // Compute grid normal for the two grid verts forming the edge
                Eigen::Vector3d g0 = gradientAt(GV, S, i, j, k, resX, resY, resZ);
                Eigen::Vector3d g1 = gradientAt(GV, S, i+1, j, k, resX, resY, resZ);
                Eigen::Vector3d grid_n = g0 + g1;

                // Compute quad normal from cell vertices
                Eigen::Vector3d pa = V.row(c0).transpose();
                Eigen::Vector3d pb = V.row(c1).transpose();
                Eigen::Vector3d pc = V.row(c2).transpose();
                Eigen::Vector3d pd = V.row(c3).transpose();
                Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
                if (quad_n.squaredNorm() < 1e-24) quad_n = (pc - pa).cross(pd - pa);

                if (grid_n.squaredNorm() > 0 && quad_n.squaredNorm() > 0) {
                    if (quad_n.dot(grid_n) < 0.0) {
                        // flip winding by swapping c1 and c3
                        std::swap(c1, c3);
                    }
                }

                quads.emplace_back(c0, c1, c2, c3);
            }
        }
    }

    // --- Edges along Y ---
    // edge between (i,j,k) and (i,j+1,k)
    for (int i = 1; i < cx; ++i) {
        for (int j = 0; j < cy; ++j) {
            for (int k = 1; k < cz; ++k) {

                int gv0 = idxV(i, j,   k);
                int gv1 = idxV(i, j+1, k);
                if (!edgeHasSignChange(gv0, gv1, iso)) continue;

                int c0 = getCellVert(i-1, j, k-1);
                int c1 = getCellVert(i,   j, k-1);
                int c2 = getCellVert(i,   j, k  );
                int c3 = getCellVert(i-1, j, k  );

                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
                Eigen::Vector3d g0 = gradientAt(GV, S, i, j, k, resX, resY, resZ);
                Eigen::Vector3d g1 = gradientAt(GV, S, i, j+1, k, resX, resY, resZ);
                Eigen::Vector3d grid_n = g0 + g1;

                Eigen::Vector3d pa = V.row(c0).transpose();
                Eigen::Vector3d pb = V.row(c1).transpose();
                Eigen::Vector3d pc = V.row(c2).transpose();
                Eigen::Vector3d pd = V.row(c3).transpose();
                Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
                if (quad_n.squaredNorm() < 1e-24) quad_n = (pc - pa).cross(pd - pa);

                if (grid_n.squaredNorm() > 0 && quad_n.squaredNorm() > 0) {
                    if (quad_n.dot(grid_n) < 0.0) {
                        std::swap(c1, c3);
                    }
                }

                quads.emplace_back(c0, c1, c2, c3);
            }
        }
    }

    // --- Edges along Z ---
    // edge between (i,j,k) and (i,j,k+1)
    for (int i = 1; i < cx; ++i) {
        for (int j = 1; j < cy; ++j) {
            for (int k = 0; k < cz; ++k) {

                int gv0 = idxV(i, j, k);
                int gv1 = idxV(i, j, k+1);
                if (!edgeHasSignChange(gv0, gv1, iso)) continue;

                int c0 = getCellVert(i-1, j-1, k);
                int c1 = getCellVert(i,   j-1, k);
                int c2 = getCellVert(i,   j,   k);
                int c3 = getCellVert(i-1, j,   k);

                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
                Eigen::Vector3d g0 = gradientAt(GV, S, i, j, k, resX, resY, resZ);
                Eigen::Vector3d g1 = gradientAt(GV, S, i, j, k+1, resX, resY, resZ);
                Eigen::Vector3d grid_n = g0 + g1;

                Eigen::Vector3d pa = V.row(c0).transpose();
                Eigen::Vector3d pb = V.row(c1).transpose();
                Eigen::Vector3d pc = V.row(c2).transpose();
                Eigen::Vector3d pd = V.row(c3).transpose();
                Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
                if (quad_n.squaredNorm() < 1e-24) quad_n = (pc - pa).cross(pd - pa);

                if (grid_n.squaredNorm() > 0 && quad_n.squaredNorm() > 0) {
                    if (quad_n.dot(grid_n) < 0.0) {
                        std::swap(c1, c3);
                    }
                }

                quads.emplace_back(c0, c1, c2, c3);
            }
        }
    }

    F.resize((int)quads.size(), 4);
    for (int f = 0; f < (int)quads.size(); ++f) {
        F.row(f) = quads[f].transpose();
    }
}


bool is_point_on_edge(const Eigen::Vector3d& p, 
                      const Eigen::Vector3d& v0, 
                      const Eigen::Vector3d& v1, 
                      double tolerance = 1e-6) 
{
    Eigen::Vector3d edge = v1 - v0;
    double edge_len_sq = edge.squaredNorm();
    
    if (edge_len_sq < 1e-12) return (p - v0).norm() < tolerance;

    double t = (p - v0).dot(edge) / edge_len_sq;

    if (t < -0.001 || t > 1.001) return false;

    Eigen::Vector3d projection = v0 + t * edge;
    double dist = (p - projection).norm();

    return dist < tolerance;
}

// Mesh extraction
void extract_mesh_from_cells(
    const std::vector<Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    std::vector<Eigen::Vector3d>& hermite_normals)
{
    // Clear output hermite_normals
    hermite_normals.clear();

    auto cellIndex = [&](int i, int j, int k) {
        return cellIndex3D(i, j, k, resX, resY, resZ);
    };

    const int cx = resX - 1;
    const int cy = resY - 1;
    const int cz = resZ - 1;

    // 1) Collect vertices (unchanged)
    std::vector<Eigen::Vector3d> verts;
    verts.reserve(cells.size());
    std::vector<int> cellVertexIndex(cells.size(), -1);

    for (int k = 0; k < cz; ++k) {
        for (int j = 0; j < cy; ++j) {
            for (int i = 0; i < cx; ++i) {
                int ci = cellIndex(i, j, k);
                const Cell& c = cells[ci];
                if (!c.has_vertex) continue;

                cellVertexIndex[ci] = (int)verts.size();
                verts.push_back(c.vertex);
            }
        }
    }

    V.resize((int)verts.size(), 3);
    for (int i = 0; i < (int)verts.size(); ++i) {
        V.row(i) = verts[i].transpose();
    }



    // 2) Helper: check if an edge between two grid vertices has a sign change
    auto edgeHasSignChange = [&](int gv0, int gv1, double iso) {
        double s0 = S(gv0) - iso;
        double s1 = S(gv1) - iso;
        // require opposite signs (and both nonzero)
        return (s0 * s1 < 0.0);
    };

    auto getCellVert = [&](int i, int j, int k) -> int {
        if (i < 0 || i >= cx ||
            j < 0 || j >= cy ||
            k < 0 || k >= cz) return -1;
        int ci = cellIndex(i, j, k);
        return cellVertexIndex[ci];
    };

    std::vector<Eigen::Vector4i> quads;
    quads.reserve(cells.size() * 3);

    // Store hermite normals per quad (i.e., per edge)
    // Initialize to zero (as many as quads will be created)
    std::vector<Eigen::Vector3d> hermite_normals_list;
    hermite_normals_list.reserve(cells.size() * 3); 

    // Helper: find the hermite normal for an edge defined by grid vertex indices gv0/gv1
    // and the canonical cell coordinate (ci, cj, ck) of the cell on the "lower" side
    // (i.e. the cell that we initially inspect). If found, push it to hermite_normals_list
    // and return true. If not found, push a zero vector and return false.
    auto find_and_push_hermite_normal_for_edge = [&](int gv0, int gv1,
                                                     int cell_i, int cell_j, int cell_k,
                                                     int resX, int resY, int resZ) -> bool {
        // compute cell index and bounds-check
        if (cell_i < 0 || cell_i >= (resX - 1) ||
            cell_j < 0 || cell_j >= (resY - 1) ||
            cell_k < 0 || cell_k >= (resZ - 1)) {
            // out of bounds, can't find
            hermite_normals_list.push_back(Eigen::Vector3d::Zero());
            return false;
        }

        int ci = cellIndex(cell_i, cell_j, cell_k);
        const Cell& cell_a = cells[ci];

        bool found = false;
        Eigen::Vector3d v0 = GV.row(gv0).transpose();
        Eigen::Vector3d v1 = GV.row(gv1).transpose();

        for (const auto& kv : cell_a.hermite_positions) {
            int edge_id = kv.first;
            const Eigen::Vector3d& pos = kv.second;
            if (is_point_on_edge(pos, v0, v1)) {
                hermite_normals_list.push_back(cell_a.hermite_normals.at(edge_id));
                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error(
                "No hermite normal found for edge at cell (" +
                std::to_string(cell_i) + "," + std::to_string(cell_j) + "," + std::to_string(cell_k) + ")");
        }

        return found;
    };

    // 3) Build quads **only** around sign–changing edges

    auto idxV = [&](int i, int j, int k) {
        return index3D(i, j, k, resX, resY, resZ);
    };

    double iso = 0.0; // or pass isoValue in as a parameter if you want flexibility

    // --- Edges along X ---
    // edge between grid verts (i,j,k) and (i+1,j,k)
    for (int i = 0; i < cx; ++i) {
        for (int j = 1; j < cy; ++j) {
            for (int k = 1; k < cz; ++k) {

                int gv0 = idxV(i,   j, k);
                int gv1 = idxV(i+1, j, k);
                if (!edgeHasSignChange(gv0, gv1, iso)) continue;

                int c0 = getCellVert(i, j-1, k-1);
                int c1 = getCellVert(i, j,   k-1);
                int c2 = getCellVert(i, j,   k  );
                int c3 = getCellVert(i, j-1, k  );

                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
                // Compute grid normal for the two grid verts forming the edge
                Eigen::Vector3d g0 = gradientAt(GV, S, i, j, k, resX, resY, resZ);
                Eigen::Vector3d g1 = gradientAt(GV, S, i+1, j, k, resX, resY, resZ);
                Eigen::Vector3d grid_n = g0 + g1;

                // Compute quad normal from cell vertices
                Eigen::Vector3d pa = V.row(c0).transpose();
                Eigen::Vector3d pb = V.row(c1).transpose();
                Eigen::Vector3d pc = V.row(c2).transpose();
                Eigen::Vector3d pd = V.row(c3).transpose();
                Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
                if (quad_n.squaredNorm() < 1e-24) quad_n = (pc - pa).cross(pd - pa);

                if (grid_n.squaredNorm() > 0 && quad_n.squaredNorm() > 0) {
                    if (quad_n.dot(grid_n) < 0.0) {
                        // flip winding by swapping c1 and c3
                        std::swap(c1, c3);
                    }
                }

                quads.emplace_back(c0, c1, c2, c3);
                // Find and push the hermite normal for this edge using the helper
                // Representative cell for this X-edge is at (i, j-1, k-1)
                find_and_push_hermite_normal_for_edge(gv0, gv1, i, j-1, k-1, resX, resY, resZ);
            }
        }
    }

    // --- Edges along Y ---
    // edge between (i,j,k) and (i,j+1,k)
    for (int i = 1; i < cx; ++i) {
        for (int j = 0; j < cy; ++j) {
            for (int k = 1; k < cz; ++k) {

                int gv0 = idxV(i, j,   k);
                int gv1 = idxV(i, j+1, k);
                if (!edgeHasSignChange(gv0, gv1, iso)) continue;

                int c0 = getCellVert(i-1, j, k-1);
                int c1 = getCellVert(i,   j, k-1);
                int c2 = getCellVert(i,   j, k  );
                int c3 = getCellVert(i-1, j, k  );

                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
                Eigen::Vector3d g0 = gradientAt(GV, S, i, j, k, resX, resY, resZ);
                Eigen::Vector3d g1 = gradientAt(GV, S, i, j+1, k, resX, resY, resZ);
                Eigen::Vector3d grid_n = g0 + g1;

                Eigen::Vector3d pa = V.row(c0).transpose();
                Eigen::Vector3d pb = V.row(c1).transpose();
                Eigen::Vector3d pc = V.row(c2).transpose();
                Eigen::Vector3d pd = V.row(c3).transpose();
                Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
                if (quad_n.squaredNorm() < 1e-24) quad_n = (pc - pa).cross(pd - pa);

                if (grid_n.squaredNorm() > 0 && quad_n.squaredNorm() > 0) {
                    if (quad_n.dot(grid_n) < 0.0) {
                        std::swap(c1, c3);
                    }
                }

                quads.emplace_back(c0, c1, c2, c3);
                // Representative cell for this Y-edge is at (i-1, j, k-1)
                find_and_push_hermite_normal_for_edge(gv0, gv1, i-1, j, k-1, resX, resY, resZ);
            }
        }
    }

    // --- Edges along Z ---
    // edge between (i,j,k) and (i,j,k+1)
    for (int i = 1; i < cx; ++i) {
        for (int j = 1; j < cy; ++j) {
            for (int k = 0; k < cz; ++k) {

                int gv0 = idxV(i, j, k);
                int gv1 = idxV(i, j, k+1);
                if (!edgeHasSignChange(gv0, gv1, iso)) continue;

                int c0 = getCellVert(i-1, j-1, k);
                int c1 = getCellVert(i,   j-1, k);
                int c2 = getCellVert(i,   j,   k);
                int c3 = getCellVert(i-1, j,   k);

                if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
                Eigen::Vector3d g0 = gradientAt(GV, S, i, j, k, resX, resY, resZ);
                Eigen::Vector3d g1 = gradientAt(GV, S, i, j, k+1, resX, resY, resZ);
                Eigen::Vector3d grid_n = g0 + g1;

                Eigen::Vector3d pa = V.row(c0).transpose();
                Eigen::Vector3d pb = V.row(c1).transpose();
                Eigen::Vector3d pc = V.row(c2).transpose();
                Eigen::Vector3d pd = V.row(c3).transpose();
                Eigen::Vector3d quad_n = (pb - pa).cross(pd - pa);
                if (quad_n.squaredNorm() < 1e-24) quad_n = (pc - pa).cross(pd - pa);

                if (grid_n.squaredNorm() > 0 && quad_n.squaredNorm() > 0) {
                    if (quad_n.dot(grid_n) < 0.0) {
                        std::swap(c1, c3);
                    }
                }

                quads.emplace_back(c0, c1, c2, c3);
                // Representative cell for this Z-edge is at (i-1, j-1, k)
                find_and_push_hermite_normal_for_edge(gv0, gv1, i-1, j-1, k, resX, resY, resZ);
            }
        }
    }

    F.resize((int)quads.size(), 4);
    for (int f = 0; f < (int)quads.size(); ++f) {
        F.row(f) = quads[f].transpose();
    }
    // Move collected hermite normals to output
    if (!hermite_normals_list.empty()) {
        hermite_normals = hermite_normals_list;
    } else {
        hermite_normals.clear();
    }
}

// Overload that also fills vertexCellIndex: vertexCellIndex->size() == V.rows()
void extract_mesh_from_cells(
    const std::vector<Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    std::vector<int>* vertexCellIndex
) {
    extract_mesh_from_cells(cells, S, GV, resX, resY, resZ, V, F);
    if (!vertexCellIndex) return;

    // Recompute same vertex ordering to map back to cells
    vertexCellIndex->clear();
    vertexCellIndex->reserve(V.rows());

    auto cellIndex = [&](int i, int j, int k) {
        return cellIndex3D(i, j, k, resX, resY, resZ);
    };
    const int cx = resX - 1;
    const int cy = resY - 1;
    const int cz = resZ - 1;

    // Build a map from cell vertex ordering used earlier: iterate cells and push c.vertex
    for (int k = 0; k < cz; ++k) {
        for (int j = 0; j < cy; ++j) {
            for (int i = 0; i < cx; ++i) {
                int ci = cellIndex(i, j, k);
                const Cell& c = cells[ci];
                if (!c.has_vertex) continue;
                vertexCellIndex->push_back(ci);
            }
        }
    }

    // If sizes mismatch, shrink or fill with -1
    if ((int)vertexCellIndex->size() != V.rows()) {
        vertexCellIndex->resize(V.rows(), -1);
    }
}


double cell_diagonal(const Eigen::MatrixXd& GV, int resX, int resY, int resZ,
    double& min_x, double& min_y, double& min_z,
    double& dx, double& dy, double& dz) {
    min_x = GV.col(0).minCoeff();
    min_y = GV.col(1).minCoeff();
    min_z = GV.col(2).minCoeff();

    dx = (GV.col(0).maxCoeff() - min_x) / (double)(resX - 1);
    dy = (GV.col(1).maxCoeff() - min_y) / (double)(resY - 1);
    dz = (GV.col(2).maxCoeff() - min_z) / (double)(resZ - 1);

    double h = std::min({dx, dy, dz});
    double dim_sqrt = std::sqrt(3.0);
    double cell_diag = h * dim_sqrt;

    return cell_diag;
}


void assign_spheres_to_cells(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    std::vector<Cell>& cells,
    int resX,
    int resY,
    int resZ,
    int batch_size,
    Eigen::MatrixXi& TriF
)
{
    if (V.rows() == 0 || F.rows() == 0) return;

    // Compute the length of the diagonal of a cell
    double min_x, min_y, min_z;
    double dx, dy, dz;
    double cell_diag = cell_diagonal(GV, resX, resY, resZ, min_x, min_y, min_z, dx, dy, dz);

    int bounded_batch_size = std::min(batch_size, (int)GV.rows());
    
    std::vector<int> sphere_batch_inds; // indices into GV, S for batch of chosen spheres. First, we fill it with all spheres with negative S, and all spheres with radius less than 3*cell_diag
    std::vector<int> inds_to_choose_randomly_from; // for later
    for (int gv = 0; gv < GV.rows(); ++gv) {
        double sdf = S(gv);
        double radius = std::abs(sdf);
        if (radius < 3.0 * cell_diag) {
            sphere_batch_inds.push_back(gv);
        }else{
            inds_to_choose_randomly_from.push_back(gv);
        }
    }
    // std::cout << "Initially selected " << sphere_batch_inds.size() << " spheres based on radius < 3*cell_diag\n";
    // then, if we don't have enough spheres, fill up to batch_size by choosing randomly from the remaining spheres
    std::vector<int> remaining_inds;
    if ((int)sphere_batch_inds.size() < bounded_batch_size) {
        int n_needed = bounded_batch_size - (int)sphere_batch_inds.size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(inds_to_choose_randomly_from.begin(), inds_to_choose_randomly_from.end(), gen);
        for (int i = 0; i < n_needed && i < (int)inds_to_choose_randomly_from.size(); ++i) {
            sphere_batch_inds.push_back(inds_to_choose_randomly_from[i]);
        }
    }
    // for debugging
    // std::cout << "Assigning " << sphere_batch_inds.size() << " spheres to cells (batch size: " << bounded_batch_size << ")\n";

    // Compute the closest point on the mesh for each grid vertex
    // Ensure faces are triangles (or edges/points) before calling libigl AABB routines
    Eigen::MatrixXd GV_batch;
    Eigen::VectorXd S_batch;
    GV_batch.resize((int)sphere_batch_inds.size(), 3);
    S_batch.resize((int)sphere_batch_inds.size());
    for (int i = 0; i < (int)sphere_batch_inds.size(); ++i) {
        int gv = sphere_batch_inds[i];
        GV_batch.row(i) = GV.row(gv);
        S_batch(i) = S(gv);
    }
    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    igl::point_mesh_squared_distance(GV_batch, V, TriF, sqrD, I, C);

    // For each grid vertex, create a sphere and assign it to the containing cell
    for (int gv = 0; gv < GV_batch.rows(); ++gv) {
    // for(int idx_in_batch : sphere_batch_inds){
        // int gv = idx_in_batch;
        double sdf = S_batch(gv);
        double radius = std::abs(sdf);
        int sign = (sdf >= 0.0) ? +1 : -1;

        double dist_to_closest = std::sqrt(sqrD(gv));
        
        // If the sphere is "floating" too far from the surface, ignore it
        if ((dist_to_closest - radius) >= cell_diag) {
            continue;
        }

        // // Ignore pseudospheres (i.e., spheres whose distance to closest point is greater than radius)
        // if (sign == -1 && dist_to_closest > radius) {
        //     continue;
        // }
        

        // Find the index of the cell containing the closest point
        int ci = std::floor((C(gv, 0) - min_x) / dx);
        int cj = std::floor((C(gv, 1) - min_y) / dy);
        int ck = std::floor((C(gv, 2) - min_z) / dz);

        // Clamp indices
        ci = std::max(0, std::min(ci, resX - 2));
        cj = std::max(0, std::min(cj, resY - 2));
        ck = std::max(0, std::min(ck, resZ - 2));

        int cell_idx = ci + (resX - 1) * (cj + (resY - 1) * ck);
        if (cell_idx < 0 || cell_idx >= (int) cells.size()) {
            continue;
        }

        Cell& cell = cells[cell_idx];
        // Use emplace_back to construct the sphere in place
        cell.assigned_spheres.emplace_back(radius, sign, GV_batch.row(gv));

        // Also store the face where the closest point lies
        // Note that this face is part of the triangulated mesh, not the original F
        int face_idx = I(gv);
        // Convert to original face index if needed
        if (F.cols() == 4) {
            face_idx = face_idx / 2;   // Integer division, since each quad became 2 triangles
        }
        cell.closest_faces.push_back(face_idx);
    }
    // for each cell, sort the assigned spheres by radius (smallest first)
    // #pragma omp parallel for schedule(static)
    for (int c = 0; c < (int)cells.size(); ++c) {
        Cell& cell = cells[c];
        std::sort(cell.assigned_spheres.begin(), cell.assigned_spheres.end(),
            [](const Sphere& a, const Sphere& b) {
                return a.radius < b.radius;
            }
        );
    }    
}

// Function that computes the distance from all assigned spheres in all cells
// to the mesh surface
double compute_total_distance_to_spheres(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& TriF,
    int resX,
    int resY,
    int resZ
) {
    const int n_grid_vertices = (int)GV.rows();
    if (n_grid_vertices == 0) return 0.0;

    Eigen::MatrixXd sphere_centers(n_grid_vertices, 3);
    Eigen::VectorXd sphere_radii(n_grid_vertices);

    // Fill centers and radii from the full grid
    #pragma omp parallel for schedule(static)
    for (int gv = 0; gv < n_grid_vertices; ++gv) {
        sphere_centers.row(gv) = GV.row(gv);
        sphere_radii(gv) = std::abs(S(gv));
    }

    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    igl::point_mesh_squared_distance(sphere_centers, V, TriF, sqrD, I, C);

    // Compute cell diagonal
    double min_x, min_y, min_z;
    double dx, dy, dz;
    double cell_diag = cell_diagonal(GV, resX, resY, resZ, min_x, min_y, min_z, dx, dy, dz);

    double total_distance = 0.0;
    #pragma omp parallel for reduction(+:total_distance) schedule(static)
    for (int i = 0; i < n_grid_vertices; ++i) {
        double dist_to_mesh = std::sqrt(sqrD(i));
        double val = std::abs(dist_to_mesh - sphere_radii(i));
        // std::cout << "Val: " << val << std::endl;
        // Do not add if val is greater than the cell diagonal
        if (val >= cell_diag) {
            continue;
        }

        total_distance += val;
    }

    return total_distance;
}

// Triangulate F choosing the diagonal that is the most orthogonal to the 
// corresponding hermite normal
void triangulate_based_on_hermite_normal(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXi& TriF,
    const std::vector<Eigen::Vector3d>& hermite_normals
) {
    if (F.cols() != 4) return; // nothing to do for triangle meshes

    TriF.resize(F.rows() * 2, 3);

    for (int q = 0; q < F.rows(); ++q) {
        int a = F(q, 0);
        int b = F(q, 1);
        int c = F(q, 2);
        int d = F(q, 3);

        // Quad vertices: a,b,c,d in order
        // The diagonals are (a,c) and (b,d)

        Eigen::Vector3d n_hermite = hermite_normals[q];

        Eigen::Vector3d d1 = (V.row(c) - V.row(a)).normalized();
        double dot1 = std::abs(d1.dot(n_hermite.normalized()));

        Eigen::Vector3d d2 = (V.row(b) - V.row(a)).normalized();
        double dot2 = std::abs(d2.dot(n_hermite.normalized()));

        if (dot1 < dot2) {
            // Choose first diagonal
            TriF.row(2 * q)     << a, b, c;
            TriF.row(2 * q + 1) << a, c, d;
        } else {
            // Choose second diagonal
            TriF.row(2 * q)     << b, c, d;
            TriF.row(2 * q + 1) << b, d, a;
        }

        // Orient triangles to match geometry
        orient_triangles_for_quad(TriF, 2 * q, 2 * q + 1, a, b, c, d, V);
    }
}


// Try flipping each quad's diagonal and keep the flip if it reduces the
// total distance from grid spheres to the mesh. This mutates TriF in-place
// when an improving flip is found.
void optimize_triangulation(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXi& TriF,
    int resX,
    int resY,
    int resZ
) {
    if (F.cols() != 4) return; // nothing to do for triangle meshes

    // Compute current total distance for the starting triangulation
    double best_total = compute_total_distance_to_spheres(S, GV, V, TriF, resX, resY, resZ);
    std::cout << "Initial total distance: " << best_total << std::endl;

    // For each quad (each quad corresponds to two consecutive rows in TriF)
    for (int q = 0; q < F.rows(); ++q) {

        // std::cout << "q: " << q << std::endl;
        // Build a candidate triangulation with the q-th quad flipped
        Eigen::MatrixXi TriF_candidate = TriF;

        int a = F(q, 0);
        int b = F(q, 1);
        int c = F(q, 2);
        int d = F(q, 3);

        // Replace the two triangle rows for this quad with the other diagonal
        // Use the same consistent winding as triangulate_v2: (b,c,d) and (b,d,a)
        TriF_candidate.row(2 * q)     << b, c, d;
        TriF_candidate.row(2 * q + 1) << b, d, a;

        // Ensure candidate triangles follow geometry-based orientation
        if (V.rows() > 0) {
            orient_triangles_for_quad(TriF_candidate, 2 * q, 2 * q + 1, a, b, c, d, V);
        }

    double cand_total = compute_total_distance_to_spheres(S, GV, V, TriF_candidate, resX, resY, resZ);
        if (cand_total < best_total) {
            // Accept the candidate
            TriF = std::move(TriF_candidate);
            best_total = cand_total;
        }
    }
}

void compute_face_cell_intersections(
    std::vector<Cell>& cells,
    int resX,
    int resY,
    int resZ,
    double weight_new_pos
){
    // Direction vectors for 6 neighbors: +X, -X, +Y, -Y, +Z, -Z
    const int dirs[6][3] = {
        {1, 0, 0}, {-1, 0, 0},
        {0, 1, 0}, {0, -1, 0},
        {0, 0, 1}, {0, 0, -1}
    };

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < resZ - 1; ++k) {
        for (int j = 0; j < resY - 1; ++j) {
            for (int i = 0; i < resX - 1; ++i) {
                int ci = i + (resX - 1) * (j + (resY - 1) * k);
                Cell& cell = cells[ci];
            
                // For each face of the cell, check neighboring cells
                // and compute intersections if both cells have vertices
                
                if (!cell.has_vertex) continue;
                Eigen::Vector3d v = cell.vertex;

                // Assert that the size of face_intersections is at most 6
                assert(cell.face_intersections.size() <= 6);

                Eigen::Vector3d min_corner = cell.corners.row(0);
                Eigen::Vector3d max_corner = cell.corners.row(6);

                for (int d = 0; d < 6; ++d) {
                    int ni = i + dirs[d][0];
                    int nj = j + dirs[d][1];
                    int nk = k + dirs[d][2];

                    // Check bounds
                    if (ni < 0 || ni >= resX - 1 ||
                        nj < 0 || nj >= resY - 1 ||
                        nk < 0 || nk >= resZ - 1) {
                        continue;
                    }

                    int nci = cellIndex3D(ni, nj, nk, resX, resY, resZ);
                    const Cell& neighbor = cells[nci];

                    if (!neighbor.has_vertex) continue;
                    Eigen::Vector3d nv = neighbor.vertex;

                    // Segment that connects the two cell vertices
                    Eigen::Vector3d segment_dir = nv - v;

                    // Compute intersection of line segment v--nv with the face plane
                    double numerator = 0.0;     // (P - v) * N 
                    double denominator = 0.0;   // d * N
                    
                    if (dirs[d][0] != 0) { // X-Face
                        // If moving +X, we hit the Max X plane (Corner 6 x-coord)
                        // If moving -X, we hit the Min X plane (Corner 0 x-coord)
                        double plane_x = (dirs[d][0] > 0) ? max_corner.x() : min_corner.x();
                        numerator   = plane_x - v.x();
                        denominator = segment_dir.x();
                    } 
                    else if (dirs[d][1] != 0) { // Y-Face
                        // If moving +Y, we hit the Max Y plane (Corner 6 y-coord)
                        // If moving -Y, we hit the Min Y plane (Corner 0 y-coord)
                        double plane_y = (dirs[d][1] > 0) ? max_corner.y() : min_corner.y();
                        numerator   = plane_y - v.y();
                        denominator = segment_dir.y();
                    } 
                    else { // Z-Face
                        // If moving +Z, we hit the Max Z plane (Corner 6 z-coord)
                        // If moving -Z, we hit the Min Z plane (Corner 0 z-coord)
                        double plane_z = (dirs[d][2] > 0) ? max_corner.z() : min_corner.z();
                        numerator   = plane_z - v.z();
                        denominator = segment_dir.z();
                    }

                    // Line is parallel to the plane -> no intersection
                    if (std::abs(denominator) < 1e-9) continue;

                    double t = numerator / denominator;

                    if (t >= -1e-6 && t <= 1.0 + 1e-6) {
                        Eigen::Vector3d intersection = v + t * segment_dir;

                        auto it = cell.face_intersections.find(d);
                        if (it == cell.face_intersections.end()) {
                            // First time assigning intersection for this face
                            cell.face_intersections[d] = intersection;
                        } else {
                            // Update existing intersection with weighted average
                            it->second = intersection * weight_new_pos +
                                         it->second * (1.0 - weight_new_pos);
                        }
                    }
                }
            }
        }
    }
}


void show_total_energy(
    const std::vector<Cell>& cells,
    bool verbose
){
    double total_energy = 0.0;
    for (const Cell& c : cells) {
        total_energy += c.energy;
    }
    if (verbose) {
        std::cout << "[contouring] Total energy: " << total_energy << std::endl;
    }
}

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
){
    if (options.verbose) {
        std::cout << "[contouring] grid = " << resX << " x " << resY << " x " << resZ
                  << ", iso = " << isoValue << std::endl;
    }

    std::vector<Cell> cells;
    std::vector<Eigen::Vector3d> hermite_normals_per_quad;

    if (options.verbose) { std::cout << "[contouring] Generating cells..." << std::endl; }
    cells = generate_cells(S, GV, resX, resY, resZ, true_sdf, true_sdf_grad);
    average_hermite_normals(cells,resX,resY,resZ, options.verbose);
    if (options.verbose) { std::cout << "[contouring] Solving QEFs..." << std::endl; }

    for (Cell& c : cells) {
        c.update();
    }

    average_hermite_normals(cells,resX,resY,resZ, options.verbose);

    if (options.verbose) { std::cout << "[contouring] Extracting mesh..." << std::endl; }

    extract_mesh_from_cells(cells, S, GV, resX, resY, resZ, V, F, hermite_normals_per_quad);

    // make all the centroid
    for( Cell& c : cells){
        if( c.has_vertex ){
            c.vertex = c.getCentroid();
        }
    }
    for (int outer_iter = 0; outer_iter < options.outer_iters; ++outer_iter) {
        if (options.verbose) {
            std::cout << "\n[contouring] Outer iteration " << (outer_iter + 1)
                      << " / " << options.outer_iters << std::endl;
        }



        // Triangulate F
        Eigen::MatrixXi TriF = triangulate_v1(F, V);

        // // Optimize the triangulation of each quad based on the distance to spheres
        // optimize_triangulation(S, GV, V, F, TriF, resX, resY, resZ);

        // Eigen::MatrixXi TriF;
        // triangulate_based_on_hermite_normal(V, F, TriF, hermite_normals_per_quad);

        if (options.verbose) { std::cout << "[contouring] Assigning spheres..." << std::endl; }
        assign_spheres_to_cells(S, GV, V, F, cells, resX, resY, resZ, options.batch_size, TriF);
        
        if (options.verbose) { std::cout << "[contouring] Computing intersections..." << std::endl; }
        compute_face_cell_intersections(cells, resX, resY, resZ, options.new_face_pos_weight);

        if (options.hermite_update && outer_iter > 0){
            if (options.verbose) { std::cout << "[contouring] Updating Hermite points and normals..." << std::endl; }
            update_hermite_points_and_normals(cells, resX, resY, resZ, options.new_hermite_normal_weight, options.new_hermite_pos_weight, true, options.verbose);
        }

        if (options.verbose) { std::cout << "[contouring] Updating vertices..." << std::endl; }

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < cells.size(); ++i) {
            Cell& c = cells[i];

            if (!c.has_vertex) continue;

            c.prev_outer_vertex = c.vertex;

            for (int inner_iter = 0; inner_iter < options.inner_iters; ++inner_iter) {
                c.prev_vertex = c.vertex;
                refine_vertex_from_face_intersections(c);
                if (!c.closest_points_info.empty()) {
                    c.minimize_qef(
                        options.mu,
                        options.dc_weight,
                        options.sphere_weight,
                        options.svd_threshold,
                        options.verbose
                    );
                }
                if((c.vertex - c.prev_vertex).norm() < 1e-6) break;
            }
            // if(outer_iter % 2 ==0){c.update();}
            c.clean();
        }
        
        // Re-extract mesh after sphere assignment
        extract_mesh_from_cells(cells, S, GV, resX, resY, resZ, V, F, hermite_normals_per_quad);

        // Sum the energy from all cells and print it
        show_total_energy(cells, options.verbose);
    }

    // If no outer iterations, we just compute the centroids and extract the mesh here
    if (options.outer_iters == 0) {
        extract_mesh_from_cells(cells, S, GV, resX, resY, resZ, V, F, hermite_normals_per_quad);
    }

    
    // Copy F, triangulate and optimize it
    Eigen::MatrixXi copyF = F;
    // F = triangulate_v1(copyF, V);
    // optimize_triangulation(S, GV, V, copyF, F, resX, resY, resZ);
    // triangulate_based_on_hermite_normal(V, copyF, F, hermite_normals_per_quad);

    // Collect all Hermite points and normals across cells, then register once
    std::vector<Eigen::Vector3d> allHermitePos;
    std::vector<Eigen::Vector3d> allHermiteNrm;
    allHermitePos.reserve(resX * resY * resZ);
    allHermiteNrm.reserve(resX * resY * resZ);
    // Also extract vertex -> cell index mapping
    std::vector<float> vertexCellIndex;

    for (const Cell& c : cells) {
        if (!c.has_vertex) continue;
        for (const auto& kv : c.hermite_positions) {
            int edge_idx = kv.first;
            const Eigen::Vector3d& pos = kv.second;
            const Eigen::Vector3d& normal = c.hermite_normals.at(edge_idx);
            allHermitePos.push_back(pos);
            allHermiteNrm.push_back(normal);

            // Get cell idx
            int cell_idx = cellIndex3D(c.ix, c.iy, c.iz, resX, resY, resZ);
            vertexCellIndex.push_back((float) cell_idx);
        }
    }
}


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
)
{
    // just forward, with null pointers → generate_cells will use finite differences
    contouring(S, GV, resX, resY, resZ,
               isoValue, V, F,
               options,
               /*true_sdf=*/nullptr,
               /*true_sdf_grad=*/nullptr);
}