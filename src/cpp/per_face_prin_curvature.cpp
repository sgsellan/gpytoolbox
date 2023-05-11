#include "per_face_prin_curvature.h"
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/avg_edge_length.h>
#include <igl/massmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/per_face_normals.h>
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/barycenter.h>
#include <igl/parallel_for.h>
#include <igl/pinv.h>
#include <Eigen/SparseCore>
#include <iostream>
using namespace std;

void per_face_prin_curvature(const Eigen::MatrixXd & V,const Eigen::MatrixXi & F, Eigen::MatrixXd & PD1, Eigen::MatrixXd & PD2, Eigen::VectorXd & PC1, Eigen::VectorXd & PC2){
    
    int n = V.rows();
    int m = F.rows();
    Eigen::MatrixXi E,uE,EF,EI;
    std::vector<std::vector<int>> A,uE2E;
    Eigen::VectorXi EMAP;
  Eigen::MatrixXd N;
    std::vector<int> b;
    Eigen::MatrixXd bary;
    igl::barycenter(V,F,bary);
    igl::adjacency_list(F,A);
   igl::per_face_normals(V,F,N);
   
   igl::unique_edge_map(F,E,uE,EMAP,uE2E);
   igl::edge_flaps(F,uE,EMAP,EF,EI);
    
    
    PC1.resize(m);
    PC2.resize(m);
    PD1.resize(m,3);
    PD2.resize(m,3);
    
    //std::cout << "Starting iteration" << std::endl;


    
    
    //for (int i=0; i<m; i++) { // For each face
    igl::parallel_for(F.rows(),[&] (const int i){
         
        // Step 1: Find unique combined indeces of all vertices of face
        // uu = V(unique(u{F(i,1)},u{F(i,2)},u{F(i,3)}),:)
        // vv = uu-bary(i,:) what is this notation in c++...
        // Triangle i has edges E.row(i), E.row(i+m), E.row(i+2m)
        // So incident (repeated) faces are EF(i,0), EF(i,1), EF(i+m,0),
        // EF(i+m,1), EF(i+2m,0) and EF(i+2m,1);
        // std::cout << "entered loop" << std::endl;
        std::vector<int> vertex_indeces,v1,v2,v3;
        std::vector<std::vector<int>> v123;
        // v123.resize(3);
        //for(int j = 0; j < 3; j++){
        //    v123[j].resize(3);
            // std::cout << "entered loop" << std::endl;
            //std::cout << EF(EMAP(i+(j*m)),0) << std::endl;
           // std::cout << EF(EMAP(i+(j*m)),1) << std::endl;
            
         //   for(int orient = 0; orient < 2; orient++){
         //       if(EF(EMAP(i+(j*m)),orient) != i){
                 //   std::cout << "found one" << std::endl;
         //       v123[j].push_back(F(EF(EMAP(i+(j*m)),orient),0));
         //       v123[j].push_back(F(EF(EMAP(i+(j*m)),orient),1));
         //       v123[j].push_back(F(EF(EMAP(i+(j*m)),orient),2));
         //       }
         //       }
          //  }
        // std::cout << "exited loop" << std::endl;
        //v1 = v123[0];
        //v2 = v123[1];
        //v3 = v123[2];
        //assert(v1.size() == 3);
        //assert(v2.size() == 3);
        //assert(v3.size() == 3);
       //  std::cout << "survived assertions" << std::endl;
        
        Eigen::MatrixXd P;
        Eigen::Vector3d w,uu,vv;
        Eigen::VectorXd u,v,b,a;
        
        double E,FF,G,e,f,g,det;
        v1 = A[F(i,0)];
        v2 = A[F(i,1)];
        v3 = A[F(i,2)];
        vertex_indeces.insert(vertex_indeces.end(),v1.begin(),v1.end());
        vertex_indeces.insert(vertex_indeces.end(),v2.begin(),v2.end());
        vertex_indeces.insert(vertex_indeces.end(),v3.begin(),v3.end());
        sort( vertex_indeces.begin(), vertex_indeces.end() );
        vertex_indeces.erase( unique( vertex_indeces.begin(), vertex_indeces.end() ), vertex_indeces.end() );
        int k = vertex_indeces.size();
        for (int j=0; j<k; j++) {
            P.conservativeResize(j+1,3);
            P.row(j) = V.row(vertex_indeces[j]) - bary.row(i);
        }
        // END OF STEP 1
        
        
        // Step 2: Build orthonormal basis from N
        w = N.row(i);

        if (w(2)==-1) {
            vv(0) = 0;
            vv(1) = -1;
            vv(2) = 0;
            uu(0) = -1;
            uu(1) = 0;
            uu(2) = 0;
        }
        else{
            vv(0) = 1-(pow(w(0),2)/(1+w(2)));
            vv(1) = -w(0)*w(1)/(1+w(2));
            vv(2) = -w(0);
            uu(0) = -w(0)*w(1)/(1+w(2));
            uu(1) = 1-(pow(w(1),2)/(1+w(2)));
            uu(2) = -w(1);
        }
        // END OF STEP 2
        // STEP 3: PROJECT
        Eigen::MatrixXd S(3,2);
        S.col(0) = uu;
        S.col(1) = vv;
        u = P*uu;
        v = P*vv;
        b = P*w;
        // END OF STEP 3
        // STEP 4: FIT QUADRIC
        Eigen::MatrixXd A2(k,6);
        Eigen::MatrixXd A2_pseudo(6,k);
        A2.col(0) = u;
        A2.col(1) = v;
        A2.col(2) = u.cwiseProduct(u);
        A2.col(3) = u.cwiseProduct(v);
        A2.col(4) = v.cwiseProduct(v);
        A2.col(5).setOnes();
        igl::pinv(A2,A2_pseudo);
        a = A2_pseudo*b;
        // END OF STEP 4
        // STEP 5: BUILD SS
        E = 1+(pow(a(0),2));
        FF = a(0)*a(1);
        G = 1+(pow(a(1),2));
        e = (2*a(2))/sqrt((pow(a(0),2))+1+(pow(a(1),2)));
        f = (a(3))/sqrt((pow(a(0),2))+1+(pow(a(1),2)));
        g = (2*a(4))/sqrt((pow(a(0),2))+1+(pow(a(1),2)));
        Eigen::Matrix2d S1,S2,SS;
        S1 << e,f,f,g;
        det = (G*E)-(pow(FF,2));
        if (det==0) {
            Eigen::Matrix2d S2_pseudo;
            S2 << E,FF,FF,G;
            igl::pinv(S2,S2_pseudo);
            SS = S1*S2_pseudo;
            //std::cout << "Det zero" << std::endl;
        }else{
            S2 << G,-FF,-FF,E;
            S2 /= det;
            SS = S1*S2;
        }
        // END OF STEP 5
        // STEP 6: EIGENALALYSIS OF SS
        // std::cout << "Before eigensolver" << std::endl;
        Eigen::EigenSolver<Eigen::MatrixXd> es(SS);
        PC1(i) = - es.eigenvalues().real().coeff(0);
        PC2(i) = - es.eigenvalues().real().coeff(1);
        if (PC1(i)>PC2(i)) {
            std::swap(PC1(i),PC2(i));
            PD1.row(i) = es.eigenvectors().real().coeff(0,1)*uu + es.eigenvectors().real().coeff(1,1)*vv;
            PD2.row(i) = es.eigenvectors().real().coeff(0,0)*uu + es.eigenvectors().real().coeff(1,0)*vv;
        }else{
            PD1.row(i) = es.eigenvectors().real().coeff(0,0)*uu + es.eigenvectors().real().coeff(1,0)*vv;
            PD2.row(i) = es.eigenvectors().real().coeff(0,1)*uu + es.eigenvectors().real().coeff(1,1)*vv;
        }
       // PD1.row(i) = uu;
       // PD2.row(i) = w;
        //std::cout << "Ending iteration" << std::endl;
    },0);
        //
    //}
    
    
}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp principal_curvatures_silvia.cpp -o main

