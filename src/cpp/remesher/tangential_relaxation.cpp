#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/avg_edge_length.h>
#include <igl/massmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/per_face_normals.h>
#include <igl/barycenter.h>
#include <igl/pinv.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
#include <Eigen/SparseCore>
#include <igl/adjacency_list.h>
#include <igl/adjacency_matrix.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/avg_edge_length.h>
#include <igl/edge_flaps.h>
#include <igl/unique_edge_map.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/principal_curvature.h>
#include <igl/collapse_edge.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/C_STR.h>
#include <igl/flip_edge.h>
#include <igl/remove_duplicate_vertices.h>
#include <limits>
using namespace std;

namespace {
    // Closest point to `p` on the segment [a,b].
    inline Eigen::RowVector3d closest_point_on_segment(
            const Eigen::RowVector3d & p,
            const Eigen::RowVector3d & a,
            const Eigen::RowVector3d & b) {
        Eigen::RowVector3d ab = b - a;
        double denom = ab.dot(ab);
        if (denom <= 0.0) return a;
        double t = (p - a).dot(ab) / denom;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        return a + t * ab;
    }
}

void tangential_relaxation(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges,
        Eigen::MatrixXd & V0 ,Eigen::MatrixXi & F0, Eigen::VectorXd & lambda,
        Eigen::MatrixXd & featV0, Eigen::MatrixXi & featE0, bool project){
    using namespace Eigen;
        MatrixXd N,V_projected,V_fixed;
        VectorXd sqrD;
        VectorXi sqrI;
        std::vector<std::vector<int>> A;
        Matrix3d NN;

        V_fixed = V;

        int n = V.rows();

        igl::adjacency_list(F,A);

        int num_feat = feature.size();

        // Fully fixed (explicit) feature vertices: never moved.
        std::vector<bool> explicit_fixed(n,false);
        for (int s = 0; s < num_feat; s++) {
            explicit_fixed[feature(s)] = true;
        }

        // Feature-edge adjacency: for each vertex on a feature edge, list its
        // feature-edge neighbors. Degree-2 vertices are "line" vertices that
        // slide along the crease; any other (nonzero) degree marks a corner /
        // junction that stays fixed.
        std::vector<std::vector<int>> feat_adj(n);
        for (int j = 0; j < feature_edges.rows(); j++) {
            int u = feature_edges(j,0);
            int v = feature_edges(j,1);
            feat_adj[u].push_back(v);
            feat_adj[v].push_back(u);
        }

        igl::per_vertex_normals(V,F,N);

        // Smooth: fixed vertices stay put, feature-line vertices get a 1D
        // Laplacian along the crease, everything else gets tangent-plane
        // smoothing.
        for(int i = 0; i < n; i++){
            int fdeg = (int)feat_adj[i].size();
            bool is_fixed = explicit_fixed[i] || (fdeg >= 1 && fdeg != 2);
            if (is_fixed) {
                continue;
            }
            if (fdeg == 2) {
                // 1D Laplacian along the feature line.
                Eigen::RowVector3d q = 0.5*(V.row(feat_adj[i][0]) + V.row(feat_adj[i][1]));
                V.row(i) = q;
                continue;
            }
            // Regular vertex: tangent-plane smoothing.
            Eigen::RowVector3d q,p;
            q.setZero();
            p.setZero();
            for(int j = 0; j < A[i].size(); j++){
                q = q + (V.row(A[i][j])/A[i].size());
            }
            NN = lambda(i)*(Eigen::MatrixXd::Identity(3,3) - N.row(i).transpose()*(N.row(i)));
            p = (V.row(i).transpose()-(NN*(V.row(i).transpose() - q.transpose()))).transpose();
            V.row(i) = p;
        }

        // Reproject regular vertices onto the original surface.
        igl::point_mesh_squared_distance(V,V0,F0,sqrD,sqrI,V_projected);

        for(int i = 0; i < n; i++){
            int fdeg = (int)feat_adj[i].size();
            bool is_fixed = explicit_fixed[i] || (fdeg >= 1 && fdeg != 2);
            if (is_fixed) {
                // Restore exact original position.
                V.row(i) = V_fixed.row(i);
                continue;
            }
            if (fdeg == 2) {
                if (project && featE0.rows() > 0) {
                    // Reproject the smoothed feature-line vertex onto the
                    // frozen original feature polyline so the crease stays sharp.
                    Eigen::RowVector3d p = V.row(i);
                    double best = std::numeric_limits<double>::infinity();
                    Eigen::RowVector3d best_pt = p;
                    for (int j = 0; j < featE0.rows(); j++) {
                        Eigen::RowVector3d a = featV0.row(featE0(j,0));
                        Eigen::RowVector3d b = featV0.row(featE0(j,1));
                        Eigen::RowVector3d c = closest_point_on_segment(p,a,b);
                        double d = (p-c).squaredNorm();
                        if (d < best) { best = d; best_pt = c; }
                    }
                    V.row(i) = best_pt;
                }
                // else: keep the smoothed (already-assigned) position.
                continue;
            }
            // Regular vertex: take the surface-projected position.
            V.row(i) = V_projected.row(i);
        }
}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main
