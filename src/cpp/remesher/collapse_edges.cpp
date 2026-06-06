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
#include <igl/is_edge_manifold.h>
#include <igl/edge_collapse_is_valid.h>
#include <igl/C_STR.h>
#include <igl/circulation.h>
#include <igl/decimate.h>
#include <igl/min_heap.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/infinite_cost_stopping_condition.h>
#include <igl/decimate_trivial_callbacks.h>
#include <igl/decimate_callback_types.h>
#include <unordered_set>
#include <array>
#include <set>
#include <utility>
using namespace std;

namespace {
    inline long long edge_key(int a, int b) {
        if (a > b) std::swap(a, b);
        return ((long long)a << 32) | (long long)(unsigned int)b;
    }
    int uf_find(std::vector<int>& parent, int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void uf_union(std::vector<int>& parent, int a, int b) {
        a = uf_find(parent,a); b = uf_find(parent,b);
        if (a != b) parent[a] = b;
    }
}

void collapse_edges(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges, Eigen::VectorXd & high, Eigen::VectorXd & low){
        using namespace Eigen;
    MatrixXi E,uE,EI,EF;
    VectorXi EMAP,I,J;
    VectorXd data;
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    std::vector<std::vector<int>> uE2E;
    std::vector<std::vector<int>> vertex_face_adjacency;
    std::vector<int> small_edges;
    int e1,e2,f1,f2,e;
    int n = V.rows();


    int num_feature = feature.size();
    std::vector<std::vector<int>> A;
    igl::adjacency_list(F,A);

    // Fixed (frozen) feature vertices: never collapsed.
    std::vector<bool> explicit_fixed;
    explicit_fixed.resize(n,false);
    for (int s = 0; s < num_feature; s++) {
        explicit_fixed[feature(s)] = true;
    }

    // Feature-edge bookkeeping: degree (number of incident feature edges) and a
    // set for fast "is this edge a feature edge" lookups.
    std::vector<int> feat_deg(n,0);
    std::unordered_set<long long> fe_set;
    for (int j = 0; j < feature_edges.rows(); j++) {
        feat_deg[feature_edges(j,0)]++;
        feat_deg[feature_edges(j,1)]++;
        fe_set.insert(edge_key(feature_edges(j,0),feature_edges(j,1)));
    }

    igl::decimate_stopping_condition_callback stopping_condition;

    igl::decimate_cost_and_placement_callback shortest_edge_and_midpoint_lambda =
        [&A,&low,&high,&explicit_fixed,&feat_deg,&fe_set](
            const int e,
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & F,
            const Eigen::MatrixXi & E,
            const Eigen::VectorXi & EMAP,
            const Eigen::MatrixXi & EF,
            const Eigen::MatrixXi & EI,
            double & cost,
            Eigen::RowVectorXd & p)
    {
        igl::shortest_edge_and_midpoint(e,V,F,E,EMAP,EF,EI,cost,p);
        const int a = E(e,0);
        const int b = E(e,1);

        // Fully fixed feature vertices are never collapsed (original behavior).
        if (explicit_fixed[a] || explicit_fixed[b]) {
            cost = std::numeric_limits<double>::infinity();
            return;
        }

        const bool fva = feat_deg[a] >= 1; // a is on a feature edge
        const bool fvb = feat_deg[b] >= 1; // b is on a feature edge
        if (fva || fvb) {
            if (fva && fvb) {
                // Both endpoints are feature vertices.
                const bool is_fe = fe_set.count(edge_key(a,b)) > 0;
                if (!is_fe) {
                    // Collapsing would merge two different feature curves.
                    cost = std::numeric_limits<double>::infinity();
                    return;
                }
                const bool corner_a = feat_deg[a] != 2; // junction / endpoint
                const bool corner_b = feat_deg[b] != 2;
                if (corner_a && corner_b) {
                    // Would remove a feature junction/corner.
                    cost = std::numeric_limits<double>::infinity();
                    return;
                }
                // Collapse along the feature: keep the result on an endpoint so
                // it stays on the feature. Prefer a corner if there is one.
                if (corner_a) p = V.row(a);
                else if (corner_b) p = V.row(b);
                else p = V.row(a);
            } else {
                // Exactly one endpoint is a feature vertex (the edge is not a
                // feature edge): collapse the regular vertex onto the feature
                // vertex, which therefore does not move.
                if (fva) p = V.row(a);
                else p = V.row(b);
            }
        }

        if ( (V.row(a)-V.row(b)).norm() > ((low(a)+low(b))/2) ) {
            cost = std::numeric_limits<double>::infinity();
            return;
        }
        for(int i = 0; i < A[b].size(); i++){
            if(!fvb && (V.row(A[b][i])-p).norm() > high(b)){
                cost = std::numeric_limits<double>::infinity();
                return;
            }
        }
        for(int r = 0; r < A[a].size(); r++){
            if(!fva && (V.row(A[a][r])-p).norm() > high(a)){
                cost = std::numeric_limits<double>::infinity();
                return;
            }
        }

	// BUILD N
	int ccw = E(e,0)>E(e,1);
std::vector<int> N;
  N.reserve(6);
  const int m = EMAP.size()/3;
  assert(m*3 == EMAP.size());
  const auto & step = [&](
    const int e,
    const int ff,
    int & ne,
    int & nf)
  {
    assert((EF(e,1) == ff || EF(e,0) == ff) && "e should touch ff");
    const int nside = EF(e,0)==ff?1:0;
    const int nv = EI(e,nside);
    // get next face
    nf = EF(e,nside);
    // get next edge
    const int dir = ccw?-1:1;
    ne = EMAP(nf+m*((nv+dir+3)%3));
  };
  // Always start with first face (ccw in step will be sure to turn right
  // direction)
  const int f0 = EF(e,0);
  int fi = f0;
  int ei = e;
  while(true)
  {
    step(ei,fi,ei,fi);
    N.push_back(fi);
    // back to start?
    if(fi == f0)
    {
      break;
    }
  }
		for(const int f : N)
	            {
	                if( f == 0 || f ==  N.size()-1)
	                {

	                    // skip
	                    continue;
	                }
	                // Grab the three corners of the face
	                Eigen::RowVector3d p_before[3], p_after[3];
	                for(int c = 0;c<3;c++)
	                {
	                    const int v = F(f,c);
	                    if( v == E(e,0) || v == E(e,1))
	                    {
	                        p_after[c] = p;
	                    }else
	                    {
	                        p_after[c] = V.row(v);
	                    }
	                    p_before[c] = V.row(v);
	                }
	                const Eigen::RowVector3d n_before =
	                ((p_before[1]- p_before[0]).cross(p_before[2]- p_before[0])).normalized();
	                const Eigen::RowVector3d n_after =
	                ((p_after[1]- p_after[0]).cross(p_after[2]- p_after[0])).normalized();
	                if( n_before.dot(n_after) < n_after.norm()/2 )
	                   {
	                       cost = std::numeric_limits<double>::infinity();
	                   }
	                   }
        };

    igl::infinite_cost_stopping_condition(shortest_edge_and_midpoint_lambda,stopping_condition);
    igl::decimate_pre_collapse_callback pre_collapse;
    igl::decimate_post_collapse_callback post_collapse_trivial;
    igl::decimate_trivial_callbacks(pre_collapse,post_collapse_trivial);

    // Record every successful collapse so we can remap feature data afterwards.
    std::vector<std::pair<int,int>> merges;
    igl::decimate_post_collapse_callback post_collapse =
        [&merges](
            const Eigen::MatrixXd & /*V*/,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & E,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & /*EF*/,
            const Eigen::MatrixXi & /*EI*/,
            const igl::min_heap<std::tuple<double,int,int>> & /*Q*/,
            const Eigen::VectorXi & /*EQ*/,
            const Eigen::MatrixXd & /*C*/,
            const int e,
            const int /*e1*/,
            const int /*e2*/,
            const int /*f1*/,
            const int /*f2*/,
            const bool collapsed)
        {
            if (collapsed) {
                merges.emplace_back(E(e,0), E(e,1));
            }
        };

    igl::decimate(V,F,shortest_edge_and_midpoint_lambda,stopping_condition,pre_collapse,post_collapse,U,G,J,I);

    // Build a map from each original (pre-collapse) vertex index to its index in
    // the output mesh, accounting for merges (so that a removed vertex maps to
    // whichever endpoint survived).
    std::vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;
    for (const auto & mp : merges) uf_union(parent, mp.first, mp.second);

    std::vector<int> birth2new(n,-1);
    for (int s = 0; s < U.rows(); s++) birth2new[I(s)] = s;
    std::vector<int> root_new(n,-1);
    for (int o = 0; o < n; o++) {
        if (birth2new[o] >= 0) root_new[uf_find(parent,o)] = birth2new[o];
    }
    for (int o = 0; o < n; o++) birth2new[o] = root_new[uf_find(parent,o)];

    // Remap high/low to the surviving vertices.
    Eigen::VectorXd high_new,low_new;
    high_new.resize(U.rows());
    low_new.resize(U.rows());
    for (int s = 0; s < U.rows(); s++) {
        high_new(s) = high(I(s));
        low_new(s) = low(I(s));
    }

    // Remap fixed feature vertices (they never collapse, so they survive).
    Eigen::VectorXi feature_new;
    feature_new.resize(num_feature);
    {
        int j = 0;
        for (int s = 0; s < num_feature; s++) {
            int nv = birth2new[feature(s)];
            if (nv >= 0) { feature_new(j) = nv; j++; }
        }
        feature_new.conservativeResize(j);
    }

    // Remap feature edges, dropping any that fully collapsed and de-duplicating.
    std::vector<std::array<int,2>> fe_new;
    std::set<long long> seen;
    for (int j = 0; j < feature_edges.rows(); j++) {
        int a = birth2new[feature_edges(j,0)];
        int b = birth2new[feature_edges(j,1)];
        if (a < 0 || b < 0 || a == b) continue;
        long long k = edge_key(a,b);
        if (seen.count(k)) continue;
        seen.insert(k);
        fe_new.push_back({a,b});
    }
    Eigen::MatrixXi feature_edges_new((int)fe_new.size(),2);
    for (int j = 0; j < (int)fe_new.size(); j++) {
        feature_edges_new(j,0) = fe_new[j][0];
        feature_edges_new(j,1) = fe_new[j][1];
    }

    V = U;
    F = G;
    high = high_new;
    low = low_new;
    feature = feature_new;
    feature_edges = feature_edges_new;
}


// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main
