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
#include <igl/circulation.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/infinite_cost_stopping_condition.h>
using namespace std;

void split_edges(Eigen::MatrixXd & V, Eigen::MatrixXi & F, Eigen::MatrixXi & E0, Eigen::MatrixXi & uE, Eigen::VectorXi & EMAP0, std::vector<std::vector<int>> & uE2E,Eigen::VectorXd & high, Eigen::VectorXd & low,const std::vector<int> & edges_to_split){
    using namespace Eigen;

    Eigen::VectorXi EMAP;
    Eigen::MatrixXi E;


    // These are the sizes *before* the splits.
    const int n = V.rows();
    const int m = F.rows();
    const int r = uE2E.size();
    const int k = uE.rows();
    const int num_edges_to_split = edges_to_split.size();
    // Used for assignments later

    // I don't really know how to get rid of this :'(
    for (int j = 0; j<r; j++) {
        int add0 = 0;
        if(uE2E[j][0]>=m){
            add0 = 2*num_edges_to_split;
        }
        if(uE2E[j][0]>=(2*m)){
            add0 = 4*num_edges_to_split;
        }
        uE2E[j][0] = uE2E[j][0]+add0;
        int add1 = 0;
        if(uE2E[j][1]>=m){
            add1 = 2*num_edges_to_split;
        }
        if(uE2E[j][1]>=(2*m)){
            add1 = 4*num_edges_to_split;
        }
        uE2E[j][1] = uE2E[j][1]+add1;
    }
    // I guess it only adds more linear time...

    // These are the sizes *after* the splits.
    int num_faces = m+2*num_edges_to_split;
    int num_vertices = n+num_edges_to_split;
    int num_uE = k+3*num_edges_to_split;
    int num_uE2E = r+3*num_edges_to_split;
    int num_E = E0.rows()+6*num_edges_to_split;
    int num_EMAP = EMAP0.size()+6*num_edges_to_split;
    // Used for indexing and labeling.

    F.conservativeResize(num_faces,3);
    V.conservativeResize(num_vertices,3);
    high.conservativeResize(num_vertices);
    low.conservativeResize(num_vertices);
    uE.conservativeResize(num_uE,2);
    std::vector<int> val;
    val.push_back(0);
    val.push_back(0);
    uE2E.resize(num_uE2E,val);

    // INITIALIZE E and EMAP!!!!!

    E.resize(num_E,2);
    E.setZero();
    EMAP.resize(num_EMAP);
    EMAP.setZero();
    //std::cout << "debug" << std::endl;
    E.block(0,0,m,2) = E0.block(0,0,m,2);
    // std::cout << "debug" << std::endl;
    E.block(num_faces,0,m,2) = E0.block(m,0,m,2);
    // std::cout << "debug" << std::endl;
    E.block(2*num_faces,0,m,2) = E0.block(2*m,0,m,2);
    // std::cout << "debug" << std::endl;
    EMAP.segment(0,m) = EMAP0.segment(0,m);
    // std::cout << "debug" << std::endl;
    EMAP.segment(num_faces,m) = EMAP0.segment(m,m);
    // std::cout << "debug" << std::endl;
    EMAP.segment(2*num_faces,m) = EMAP0.segment(2*m,m);


    for (int i = 0; i<num_edges_to_split; i++) {



        int uei = edges_to_split[i];
        assert(uE2E[uei].size()==2);
        //          v1
        //          /|\
        //         / | \
        //     v3 /f1|f0\ v4
        //        \  |  /
        //         \ | /
        //          \|/
        //          v2

        int e0 = uE2E[uei][0];
        int e1 = uE2E[uei][1];

        int f0 = e0 % num_faces;
        int f1 = e1 % num_faces;
        int c0 = e0 / num_faces;
        int c1 = e1 / num_faces;

        // Order of edges: anti-clockwise starting at v1
        int e2 = f1+((c1+1)%3)*num_faces;
        int e3 = f1+((c1+2)%3)*num_faces;
        int e4 = f0+((c0+1)%3)*num_faces;
        int e5 = f0+((c0+2)%3)*num_faces;

        int ue2 = EMAP(e2);
        int ue3 = EMAP(e3);
        int ue4 = EMAP(e4);
        int ue5 = EMAP(e5);
//        std::cout << EMAP(e0) << std::endl;
//        std::cout << EMAP(e1) << std::endl;
//        std::cout << uei << std::endl;
        assert(EMAP(e0)==uei);
        assert(EMAP(e1)==uei);

        int v1 = F(f0, (c0+1)%3);
        int v2 = F(f0, (c0+2)%3);
        int v4 = F(f0, c0);
        int v3 = F(f1, c1);

        // Assertions for debugging
        // I think I can actually just "imagine" E existing? Actually no.
//        std::cout << "break" << std::endl;
//        std::cout << uei << std::endl;
//        std::cout << e2 << std::endl;
//        std::cout << v1 << std::endl;
//        std::cout << E(e2,0) << std::endl;
//        std::cout << E(e2,1) << std::endl;
        assert(E(e2,0)==v1);
        assert(E(e2,1)==v3);
        assert(E(e3,0)==v3);
        assert(E(e3,1)==v2);
        assert(E(e4,0)==v2);
        assert(E(e4,1)==v4);
        assert(E(e5,0)==v4);
        assert(E(e5,1)==v1);
        assert(E(e0,0)==v1);
        assert(E(e0,1)==v2);
        assert(E(e1,0)==v2);
        assert(E(e1,1)==v1);



        assert(c0 < 3);
        assert(c1 < 3);

        assert(f0 != f1);

        // *** UPDATE V ***

        // naïve: use mid-point
        V.row(n+i) = (V.row(v1)+V.row(v2))/2;
        high(n+i) = (high(v1)+high(v2))/2;
        low(n+i) = (low(v1)+low(v2))/2;
        // *** UPDATE F ***

        // f0

        F(f0,c0) = v4; // redundant
        F(f0,(c0+1)%3) = v1;
        F(f0,(c0+2)%3) = n+i; // new vertex
        // f1
        F(f1,c1) = v3; // redundant
        F(f1,(c1+1)%3) = n+i;
        F(f1,(c1+2)%3) = v1;
        // fm
        F(m+(2*i),0) = v2;
        F(m+(2*i),1) = n+i;
        F(m+(2*i),2) = v3;
        // fm+1
        F(m+(2*i)+1,0) = v2;
        F(m+(2*i)+1,1) = v4;
        F(m+(2*i)+1,2) = n+i;

        // *** UPDATE UE ***

        // uei
        uE(uei,0) = n+i;
        uE(uei,1) = v1;
        // k
        uE(k+(3*i),0) = n+i;
        uE(k+(3*i),1) = v3;
        // k+1
        uE(k+(3*i)+1,0) = n+i;
        uE(k+(3*i)+1,1) = v4;
        // k+2
        uE(k+(3*i)+2,0) = n+i;
        uE(k+(3*i)+2,1) = v2;


        // *** UPDATE uE2E ***
//        for (int j = 0; j<uE2E.size(); j++) {
//        std::cout << uE2E[j][0] << std::endl;
//        std::cout << uE2E[j][1] << std::endl;
//        }
        // uei
        uE2E[uei][0] = e0;
        uE2E[uei][1] = e1;
        // k

        uE2E[k+(3*i)][0] = f1 + ((c1+2)%3)*num_faces;
        uE2E[k+(3*i)][1] = m+(2*i) + 0*num_faces;

        // k+1
        uE2E[k+(3*i)+1][0] = f0 + ((c0+1)%3)*num_faces;
        uE2E[k+(3*i)+1][1] = m+(2*i)+1 + 0*num_faces;
        // k+2
        uE2E[k+(3*i)+2][0] = m+(2*i) + 2*num_faces;
        uE2E[k+(3*i)+2][1] = m+(2*i)+1 + 1*num_faces;
        // ue2

        if(uE2E[ue2][0]==e2){
            uE2E[ue2][0] = f1 + ((c1+1)%3)*num_faces;
        }else{
            assert(uE2E[ue2][1]==e2);
            uE2E[ue2][1] = f1 + ((c1+1)%3)*num_faces;
        }
        // ue3
        if(uE2E[ue3][0]==e3){
            uE2E[ue3][0] = m+(2*i) + 1*num_faces;
        }else{
            assert(uE2E[ue3][1]==e3);
            uE2E[ue3][1] = m+(2*i) + 1*num_faces;
        }
        // ue4
        if(uE2E[ue4][0]==e4){
            uE2E[ue4][0] = m+(2*i)+1 + 2*num_faces;
        }else{
            assert(uE2E[ue4][1]==e4);
            uE2E[ue4][1] = m+(2*i)+1 + 2*num_faces;
        }
        // ue5
        if(uE2E[ue5][0]==e5){
            uE2E[ue5][0] = f0 + ((c0+2)%3)*num_faces;
        }else{
            assert(uE2E[ue5][1]==e5);
            uE2E[ue5][1] = f0 + ((c0+2)%3)*num_faces;
        }
        // *** UPDATE E ***
        // edges in f0

        E(f0 + ((c0)%3)*num_faces,0) = v1;
        E(f0 + ((c0)%3)*num_faces,1) = n+i;
        E(f0 + ((c0+1)%3)*num_faces,0) = n+i;
        E(f0 + ((c0+1)%3)*num_faces,1) = v4;
        E(f0 + ((c0+2)%3)*num_faces,0) = v4;
        E(f0 + ((c0+2)%3)*num_faces,1) = v1;
        // edges in f1
        E(f1 + ((c1)%3)*num_faces,0) = n+i;
        E(f1 + ((c1)%3)*num_faces,1) = v1;
        E(f1 + ((c1+1)%3)*num_faces,0) = v1;
        E(f1 + ((c1+1)%3)*num_faces,1) = v3;
        E(f1 + ((c1+2)%3)*num_faces,0) = v3;
        E(f1 + ((c1+2)%3)*num_faces,1) = n+i;
        // edges in fm
        E(m+(2*i) + 0*num_faces,0) = n+i;
        E(m+(2*i) + 0*num_faces,1) = v3;
        E(m+(2*i) + 1*num_faces,0) = v3;
        E(m+(2*i) + 1*num_faces,1) = v2;
        E(m+(2*i) + 2*num_faces,0) = v2;
        E(m+(2*i) + 2*num_faces,1) = n+i;
        // edges in fm+1
        E(m+(2*i)+1 + 0*num_faces,0) = v4;
        E(m+(2*i)+1 + 0*num_faces,1) = n+i;
        E(m+(2*i)+1 + 1*num_faces,0) = n+i;
        E(m+(2*i)+1 + 1*num_faces,1) = v2;
        E(m+(2*i)+1 + 2*num_faces,0) = v2;
        E(m+(2*i)+1 + 2*num_faces,1) = v4;

        // *** UPDATE EMAP ***
        // edges in f0
        EMAP(f0 + ((c0)%3)*num_faces) = uei;
        EMAP(f0 + ((c0+1)%3)*num_faces) = k+(3*i)+1;
        EMAP(f0 + ((c0+2)%3)*num_faces) = ue5;
        // edges in f1
        EMAP(f1 + ((c1)%3)*num_faces) = uei;
        EMAP(f1 + ((c1+1)%3)*num_faces) = ue2;
        EMAP(f1 + ((c1+2)%3)*num_faces) = k+(3*i);
        // edges in fm
        EMAP(m+(2*i) + 0*num_faces) = k+(3*i);
        EMAP(m+(2*i) + 1*num_faces) = ue3;
        EMAP(m+(2*i) + 2*num_faces) = k+(3*i)+2;
        // edges in fm+1
        EMAP(m+(2*i)+1 + 0*num_faces) = k+(3*i)+1;
        EMAP(m+(2*i)+1 + 1*num_faces) = k+(3*i)+2;
        EMAP(m+(2*i)+1 + 2*num_faces) = ue4;

    }





    E0 = E;
    EMAP0 = EMAP;
}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main

