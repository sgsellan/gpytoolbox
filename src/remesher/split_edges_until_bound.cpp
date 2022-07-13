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
#include "split_edges.h"
using namespace std;

void split_edges_until_bound(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::VectorXd & high, Eigen::VectorXd & low){

    using namespace Eigen;
    int m = F.rows();
    int n = V.rows();
    int num_feat = feature.size();
    std::vector<std::vector<int>> A;
    std::vector<bool> is_feature_vertex;
    is_feature_vertex.resize(n);
    Eigen::VectorXi is_feature_vertex_vec;
    is_feature_vertex_vec.setZero(n);
    igl::adjacency_list(F,A);
    Eigen::MatrixXi E,uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<int>> uE2E;
    igl::unique_edge_map(F,E,uE,EMAP,uE2E);
    int k = uE.rows();
    //std::cout << "Start split_edges_until_bound" << std::endl;


    for (int s = 0; s < num_feat; s++) {
        is_feature_vertex[feature(s)] = true;
       // is_feature_vertex_vec(feature(s)) = 1;
    }

    bool keep_splitting = true;
    std::vector<int> edges_to_split;

    while (keep_splitting) {
        //std::cout << "A" << std::endl;
        edges_to_split.resize(0);

        for (int i = 0; i < uE.rows(); i++) {
            //std::cout << "B" << std::endl;
            if (!is_feature_vertex[uE(i,0)] && !is_feature_vertex[uE(i,1)] && uE2E[i].size()==2) {
            //if (is_feature_vertex_vec(uE(i,0))==0 && is_feature_vertex_vec(uE(i,1))==0 && uE2E[i].size()==2) {
                if ( (V.row(uE(i,0))-V.row(uE(i,1))).norm()>((high(uE(i,0))+high(uE(i,1)))/2)  ){
                    edges_to_split.push_back(i);
                    //std::cout << "C" << std::endl;
                }
            }
        }

        //std::cout << "B" << std::endl;

        //std::cout << "D" << std::endl;
        if(edges_to_split.size()==0){
            keep_splitting = false;
        }else{
            // SPLIT EDGES IN VECTOR edges_to_split
            //

            //std::cout << "Before call to split_edges" << std::endl;
            //std::cout << edges_to_split.size() << std::endl;
            split_edges(V,F,E,uE,EMAP,uE2E,high,low,edges_to_split);
            //igl::writeOBJ("test.obj",V,F);
            //igl::unique_edge_map(F,E,uE,EMAP,uE2E);
            //std::cout << igl::is_edge_manifold(F) << std::endl;
            //std::cout << "After call to split_edges" << std::endl;

        }


       keep_splitting = false; // THIS IS A PATCH, NOT GOOD
    }


}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main

