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
using namespace std;

void tangential_relaxation(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature,
        Eigen::MatrixXd & V0 ,Eigen::MatrixXi & F0, Eigen::VectorXd & lambda){
    using namespace Eigen;
        MatrixXd Q,P,N,V_projected,V_fixed;
        VectorXd dblA,sqrD;
        VectorXi sqrI;
        std::vector<std::vector<int>> A;
        Matrix3d I, NN;
        I.setIdentity();
        Eigen::MatrixXd SV;
        Eigen::MatrixXi SVI,SVJ;






        V_fixed = V;

        int n = V.rows();
        int m = F.rows();

        //igl::doublearea(V,F,dblA);

        //std::vector<double> vertex_areas;
        //vertex_areas.setZero(m);


        //for (int j = 0; j < m; j++) {
        //    vertex_areas[F(j,0)] = vertex_areas[F(j,0)] + (abs(dblA(j))/6);
        //    vertex_areas[F(j,1)] = vertex_areas[F(j,1)] + (abs(dblA(j))/6);
        //    vertex_areas[F(j,2)] = vertex_areas[F(j,2)] + (abs(dblA(j))/6);
        //}


        Eigen::MatrixXd N_before,N_after;
        igl::adjacency_list(F,A);


        int num_feat = feature.size();
        std::vector<bool> is_feature_vertex;
        is_feature_vertex.resize(n);

        for (int s = 0; s < num_feat; s++) {
            is_feature_vertex[feature(s)] = true;
        }

        Q.resize(n,3);
        P.resize(n,3);
        //           Eigen::MatrixXd N;
        igl::per_vertex_normals(V,F,N);

        for(int i = 0; i < n; i++){
            bool is_feature = is_feature_vertex[i];
            if (!is_feature) {

            Eigen::RowVector3d q,p;
            q.setZero();
            p.setZero();
            double denominator = 0.0;
            for(int j = 0; j < A[i].size(); j++){
                q = q + (V.row(A[i][j])/A[i].size());
                // q = q + (V.row(A[i][j])*vertex_areas[A[i][j]]);
                // std::cout << q << std::endl;
                // denominator = denominator + vertex_areas[A[i][j]];
                } // q is )( barycenter?
            // q = q/denominator;
            // N.row(i) = N.row(i)/N.row(i).norm();
            NN = lambda(i)*(Eigen::MatrixXd::Identity(3,3) - N.row(i).transpose()*(N.row(i)));
             p = (V.row(i).transpose()-(NN*(V.row(i).transpose() - q.transpose()))).transpose();
            // p = q;
             // std::cout << N.row(i) << std::endl;

            V.row(i) = p;

            // igl::per_face_normals(V_projected,F,Eigen::Vector3d(0,0,0),N_after);
    //            for (int j = 0; j < m ; j++) {
    //                if (N_before.row(j).dot(N_after.row(j)) < 0) {
    //                    // std::cout << "Avoided face flipping, I think." << std::endl;
    //                    V.row(i) = V_fixed.row(i);
    //                }
    //            }

            }
        }
//        igl::remove_duplicate_vertices(V,0,SV,SVI,SVJ);
//        std::cout << V.rows()-SV.rows() << std::endl;
//	igl::writeOBJ("pre-project.obj",V,F);
//
	igl::point_mesh_squared_distance(V,V0,F0,sqrD,sqrI,V_projected);
//
//
    V = V_projected;
//	igl::writeOBJ("post-project.obj",V,F);
//    igl::remove_duplicate_vertices(V,0,SV,SVI,SVJ);
//    std::cout << V.rows()-SV.rows() << std::endl;
 //   std::cout << "not projecting!" << std::endl;
}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main

