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
#include <igl/C_STR.h>
#include <igl/flip_edge.h>
using namespace std;

void equalize_valences(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature){
    using namespace igl;
    using namespace Eigen;
    VectorXd p;
    std::vector<bool> is_feature_vertex;
    Eigen::MatrixXi E,uE,EI,EF;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<int>> uE2E,A;
    igl::unique_edge_map(F,E,uE,EMAP,uE2E);



    int m = F.rows();
    int n = V.rows();
    int a,b,c,d;
    igl::adjacency_list(E,A);
    VectorXi vertex_valences;
//
    vertex_valences.setZero(n);

   // std::cout << "A" << std::endl;
    for(int j = 0; j < m; j++){
        vertex_valences(F(j,0)) = vertex_valences(F(j,0))+1;
        vertex_valences(F(j,1)) = vertex_valences(F(j,1))+1;
        vertex_valences(F(j,2)) = vertex_valences(F(j,2))+1;
    }
 //   std::cout << vertex_valences << std::endl;

//
    int k = uE.rows();
    int num_feat = feature.size();
    is_feature_vertex.resize(n);

   // std::cout << "B" << std::endl;

    for (int s = 0; s < num_feat; s++) {
        is_feature_vertex[feature(s)] = true;
    }


//
  //  std::cout << "C" << std::endl;

    std::function<void(
            Eigen::MatrixXi &, //F
            Eigen::MatrixXi &, //E
            Eigen::MatrixXi &, //uE
            Eigen::VectorXi &, //EMAP
            std::vector<std::vector<int>>  &, //uE2E
            int &)> flip_edge_adjacency = [&vertex_valences,&V,&A](
            Eigen::MatrixXi & F, //F
            Eigen::MatrixXi & E, //E
            Eigen::MatrixXi & uE, //uE
            Eigen::VectorXi & EMAP, //EMAP
            std::vector<std::vector<int>>  & uE2E, //uE2E
            int & uei)->void{
      //  std::cout << "Lambda call" << std::endl;
        int num_faces = F.rows();
        auto& half_edges = uE2E[uei];
        int f1 = half_edges[0] % num_faces;
        int f2 = half_edges[1] % num_faces;
        int c1 = half_edges[0] / num_faces;
        int c2 = half_edges[1] / num_faces;
        assert(c1 < 3);
        assert(c2 < 3);



        assert(f1 != f2);
        int v1 = F(f1, (c1+1)%3);
        int v2 = F(f1, (c1+2)%3);
        int v4 = F(f1, c1);
        int v3 = F(f2, c2);
        assert(F(f2, (c2+2)%3) == v1);
        assert(F(f2, (c2+1)%3) == v2);
        // Assert new triangle's area is nonzero
          // f1_new
          double a1 = (V.row(v1)-V.row(v3)).norm();
          double b1 = (V.row(v1)-V.row(v4)).norm();
          double c11 = (V.row(v4)-V.row(v3)).norm();
          double s1 = (a1+b1+c11)/2;
          double area_1_squared = s1*(s1-a1)*(s1-b1)*(s1-c11);
          // f2_new
          double a2 = (V.row(v2)-V.row(v3)).norm();
          double b2 = (V.row(v2)-V.row(v4)).norm();
          double c22 = (V.row(v4)-V.row(v3)).norm();
          double s2 = (a2+b2+c22)/2;
          double area_2_squared = s2*(s2-a2)*(s2-b2)*(s2-c22);
          bool bad = false;

          Eigen::RowVector3d v21,v31,v41,v24,v34,v43,v23,normf10,normf11,
                  normf12,normf20,normf21,normf22;
          v21 = V.row(v2)-V.row(v1);
          v31 = V.row(v3)-V.row(v1);
          v41 = V.row(v4)-V.row(v1);
          v24 = V.row(v2)-V.row(v4);
          v34 = V.row(v3)-V.row(v4);
          v43 = V.row(v4)-V.row(v3);
          v23 = V.row(v2)-V.row(v3);
          normf10 = v21.cross(v31);
          normf11 = v41.cross(v31);
          normf12 = v24.cross(v34);
          normf20 = v41.cross(v21);
          normf21 = v43.cross(v23);
          normf22 = v41.cross(v31);
          normf10.normalize();
          normf11.normalize();
          normf12.normalize();
          normf20.normalize();
          normf21.normalize();
          normf22.normalize();

          if (normf10.dot(normf11) < normf11.norm()/2 || normf10.dot(normf12) < normf12.norm()/2) {
              bad = true;
          }
          if (normf20.dot(normf21) < normf21.norm()/2 || normf20.dot(normf22) < normf22.norm()/2) {
              bad = true;
          }

        if (area_1_squared == 0) {
            bad = true;
        }
        if (area_2_squared == 0) {
            bad = true;
        }

                if(std::count(A[v3].begin(),A[v3].end(),v4)){
                    bad = true; // is it gonna generate non-manifold??
                }

        if(uE2E[uei].size() != 2){
            bad = true;
            }


         if (!bad){
             //std::cout << "D1" << std::endl;
        igl::flip_edge(F,E,uE,EMAP,uE2E,uei);
             //std::cout << "D2" << std::endl;
        //std::cout << "Lambda call test" << std::endl;
        assert(uE(uei,0)==v3);
        assert(uE(uei,1)==v4);

      //  std::cout << "updating_valences" << std::endl;
        vertex_valences(v1) = vertex_valences(v1)-1;
        vertex_valences(v2) = vertex_valences(v2)-1;
        vertex_valences(v3) = vertex_valences(v3)+1;
        vertex_valences(v4) = vertex_valences(v4)+1;

             std::remove_if(A[v1].begin(),A[v1].begin(),[&v2](const int & v){return v==v2;});
             std::remove_if(A[v2].begin(),A[v2].begin(),[&v1](const int & v){return v==v1;});
             A[v3].push_back(v4);
             A[v4].push_back(v3);

         }
        //std::cout << "Lambda call end" << std::endl;
    };










    //std::cout << "D" << std::endl;


    for(int i = 0; i < k; i++){
        //std::cout << uE2E[i].size() << std::endl;
        if(uE2E[i].size()!=2){
            continue;
        }
        bool is_feature_edge = false;
        int f1 = uE2E[i][0] % m;
        int f2 = uE2E[i][1] % m;
        int c1 = uE2E[i][0] / m;
        int c2 = uE2E[i][1] / m;
//        std::cout << f1 << std::endl;
//        std::cout << f2 << std::endl;
//        std::cout << c1 << std::endl;
//        std::cout << c2 << std::endl;
        int a = F(f1, (c1+1)%3);
        int b = F(f1, (c1+2)%3);
        int c = F(f1, c1);
        int d = F(f2, c2);
        if (is_feature_vertex[a] || is_feature_vertex[b] || is_feature_vertex[c] || is_feature_vertex[d]) {
            is_feature_edge = true;
        }
        //std::cout << "E" << std::endl;

        if(!is_feature_edge){
            // FIND VALENCES
            int deviation_pre = abs(vertex_valences(a)-6)+
                    abs(vertex_valences(b)-6)+
                    abs(vertex_valences(c)-6)+
                    abs(vertex_valences(d)-6);
//std::cout << i << std::endl;
            // igl::flip_edge(V,F,E,uE,EMAP,uE2E,i);
            int deviation_post = abs(vertex_valences(a)-1-6)+
                    abs(vertex_valences(b)-6-1)+
                    abs(vertex_valences(c)-6+1)+
                    abs(vertex_valences(d)-6+1);
            // std::cout << i << std::endl;
       //     std::cout << deviation_pre << std::endl;
       //     std::cout << deviation_post << std::endl;

            if(deviation_pre > deviation_post){
                flip_edge_adjacency(F,E,uE,EMAP,uE2E,i);
            }

        }

        //std::cout << uE.row(i+1) << std::endl;
    }
//    std::cout << flipped << std::endl;
//    std::cout << k << std::endl;
//




    // PLACEHOLDER
}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main

