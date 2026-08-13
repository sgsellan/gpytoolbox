#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/avg_edge_length.h>
#include <igl/massmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/per_face_normals.h>
#include <igl/barycenter.h>
#include <igl/pinv.h>
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
#include <igl/writeOBJ.h>
#include <igl/C_STR.h>
#include <igl/circulation.h>
#include <igl/is_edge_manifold.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/infinite_cost_stopping_condition.h>
#include <unordered_map>
#include <array>
#include <utility>
#include "split_edges.h"
using namespace std;

namespace {
    // Key for an unordered edge (pair of vertex indices).
    inline long long edge_key(int a, int b) {
        if (a > b) std::swap(a, b);
        return ((long long)a << 32) | (long long)(unsigned int)b;
    }
}

void split_edges_until_bound(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges, Eigen::VectorXd & high, Eigen::VectorXd & low){

    using namespace Eigen;
    int m = F.rows();
    int n = V.rows();
    int num_feat = feature.size();
    std::vector<std::vector<int>> A;
    std::vector<bool> is_feature_vertex;
    is_feature_vertex.resize(n);
    igl::adjacency_list(F,A);
    Eigen::MatrixXi E,uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<int>> uE2E;
    igl::unique_edge_map(F,E,uE,EMAP,uE2E);

    for (int s = 0; s < num_feat; s++) {
        is_feature_vertex[feature(s)] = true;
    }

    // Map each feature edge (as an unordered vertex pair) to its row in
    // feature_edges so that we can update it when it is split.
    std::unordered_map<long long,int> feature_edge_row;
    for (int j = 0; j < feature_edges.rows(); j++) {
        feature_edge_row[edge_key(feature_edges(j,0),feature_edges(j,1))] = j;
    }

    bool keep_splitting = true;
    std::vector<int> edges_to_split;

    while (keep_splitting) {
        edges_to_split.resize(0);

        for (int i = 0; i < uE.rows(); i++) {
            // A feature edge is always eligible to be split; otherwise the
            // edge must not touch a fixed feature vertex (those freeze their
            // one-ring, as in the original algorithm).
            bool is_feature_edge = feature_edge_row.count(edge_key(uE(i,0),uE(i,1)))>0;
            bool blocked = (is_feature_vertex[uE(i,0)] || is_feature_vertex[uE(i,1)]) && !is_feature_edge;
            if (!blocked && uE2E[i].size()==2) {
                if ( (V.row(uE(i,0))-V.row(uE(i,1))).norm()>((high(uE(i,0))+high(uE(i,1)))/2)  ){
                    edges_to_split.push_back(i);
                }
            }
        }

        if(edges_to_split.size()==0){
            keep_splitting = false;
        }else{
            // Record the endpoints (and feature status) of each edge before the
            // split. split_edges assigns the new midpoint of the i-th edge in
            // edges_to_split to vertex index (n_before + i).
            const int n_before = V.rows();
            std::vector<int> split_v0(edges_to_split.size()), split_v1(edges_to_split.size());
            std::vector<int> split_feat_row(edges_to_split.size(), -1);
            for (size_t i = 0; i < edges_to_split.size(); i++) {
                int uei = edges_to_split[i];
                split_v0[i] = uE(uei,0);
                split_v1[i] = uE(uei,1);
                auto it = feature_edge_row.find(edge_key(uE(uei,0),uE(uei,1)));
                if (it != feature_edge_row.end()) {
                    split_feat_row[i] = it->second;
                }
            }

            split_edges(V,F,E,uE,EMAP,uE2E,high,low,edges_to_split);

            // Update feature_edges: replace each split feature edge (v0,v1) by
            // the two halves (v0,w) and (w,v1), where w is the new midpoint.
            std::vector<std::array<int,2>> fe(feature_edges.rows());
            for (int j = 0; j < feature_edges.rows(); j++) {
                fe[j] = {feature_edges(j,0), feature_edges(j,1)};
            }
            for (size_t i = 0; i < edges_to_split.size(); i++) {
                if (split_feat_row[i] < 0) continue;
                int w = n_before + (int)i;
                int row = split_feat_row[i];
                fe[row] = {split_v0[i], w};
                fe.push_back({w, split_v1[i]});
            }
            feature_edges.resize((int)fe.size(),2);
            for (int j = 0; j < (int)fe.size(); j++) {
                feature_edges(j,0) = fe[j][0];
                feature_edges(j,1) = fe[j][1];
            }
        }

       keep_splitting = false; // THIS IS A PATCH, NOT GOOD
    }


}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main
