#include "equalize_valences.h"
#include "collapse_edges.h"
#include "tangential_relaxation.h"
#include <igl/is_edge_manifold.h>
#include <igl/writeOBJ.h>
#include "split_edges_until_bound.h"
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/circulation.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/avg_edge_length.h>
#include <iostream>

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature, bool project){
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F0;

    Eigen::VectorXd high,low,lambda;
    high = 1.4*target;
    low = 0.7*target;

	F0 = F;
	V0 = V;
    // Iterate the four steps
    for (int i = 0; i<iters; i++) {
    	split_edges_until_bound(V,F,feature,high,low); // Split
    	collapse_edges(V,F,feature,high,low); // Collapse
    	equalize_valences(V,F,feature); // Flip
    	int n = V.rows();
    	lambda = Eigen::VectorXd::Constant(n,1.0);
	if(!project){
		V0 = V;
		F0 = F;
	}
	tangential_relaxation(V,F,feature,V0,F0,lambda); // Relax
    }
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature){
remesh_botsch(V,F,target,iters,feature,false);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters){
	Eigen::VectorXi feature;
	feature.resize(0);
	remesh_botsch(V,F,target,iters,feature);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, bool project){
	Eigen::VectorXi feature;
	feature.resize(0);
	remesh_botsch(V,F,target,iters,feature,project);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target){
	int iters = 10;
	remesh_botsch(V,F,target,iters);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters){
	Eigen::VectorXi feature;
	feature.resize(0);
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters,feature);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, bool project){
	Eigen::VectorXi feature;
	feature.resize(0);
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters,feature,project);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double){
	int iters = 10;
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F){
	double h = igl::avg_edge_length(V,F);
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,h);
	remesh_botsch(V,F,target);
}
// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main

