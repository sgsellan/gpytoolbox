#include "octree.h"
#include <igl/octree.h>
#include <igl/list_to_matrix.h>
#include <igl/unique_rows.h>

// What is inside the octree:
// Eigen::MatrixXi CH; // children indeces
// Eigen::MatrixXd CN; // centers
// Eigen::VectorXd W;  // length of box

Octree::Octree(Eigen::MatrixXd P){
    igl::octree(P,point_indeces,CH,CN,W);
    // Eigen::MatrixXi all_quads;
    // Eigen::MatrixXd all_verts;
    std::vector<std::vector<double> >  v_list;
    std::vector<std::vector<int> > q_list;
    for (int i = 0; i < CN.rows(); i++)
    {
        bool is_child = true;
        for (int s = 0; s < 8; s++)
        {
            if (CH(i,s)>=0)
            {
                is_child = false;
            }
        }

        if (is_child)
            {     
            Eigen::RowVector3d min_corner = CN.row(i) - 0.5*W(i)*Eigen::RowVector3d(1,1,1);
            Eigen::RowVector3d max_corner = CN.row(i) + 0.5*W(i)*Eigen::RowVector3d(1,1,1);
            int n = v_list.size();
            v_list.push_back({min_corner(0),min_corner(1),min_corner(2)});
            v_list.push_back({min_corner(0),max_corner(1),min_corner(2)});
            v_list.push_back({max_corner(0),min_corner(1),min_corner(2)});
            v_list.push_back({max_corner(0),max_corner(1),min_corner(2)});
            v_list.push_back({max_corner(0),min_corner(1),max_corner(2)});
            v_list.push_back({max_corner(0),max_corner(1),max_corner(2)});
            v_list.push_back({min_corner(0),min_corner(1),max_corner(2)});
            v_list.push_back({min_corner(0),max_corner(1),max_corner(2)});
            q_list.push_back({n+0,n+1,n+2,n+3,n+4,n+5,n+6,n+7});
        }
        igl::list_to_matrix(v_list,all_verts);
        igl::list_to_matrix(q_list,all_quads);
    }
};

void Octree::get_edges(Eigen::MatrixXd & EV, Eigen::MatrixXi & EI){
std::vector<std::vector<double> >  vEV;
  std::vector<std::vector<int> > vEE;
  for(int i = 0; i<CN.rows(); i++)
  {
    int n = vEV.size();
    vEE.push_back({n+0,n+1});
    vEE.push_back({n+0,n+2});
    vEE.push_back({n+0,n+6});
    vEE.push_back({n+1,n+3});
    vEE.push_back({n+1,n+7});
    vEE.push_back({n+2,n+3});
    vEE.push_back({n+2,n+4});
    vEE.push_back({n+3,n+5});
    vEE.push_back({n+4,n+5});
    vEE.push_back({n+4,n+6});
    vEE.push_back({n+5,n+7});
    vEE.push_back({n+6,n+7});
    Eigen::RowVector3d min_corner = CN.row(i) - 0.5*W(i)*Eigen::RowVector3d(1,1,1);
    Eigen::RowVector3d max_corner = CN.row(i) + 0.5*W(i)*Eigen::RowVector3d(1,1,1);
    vEV.push_back({min_corner(0),min_corner(1),min_corner(2)});
    vEV.push_back({min_corner(0),max_corner(1),min_corner(2)});
    vEV.push_back({max_corner(0),min_corner(1),min_corner(2)});
    vEV.push_back({max_corner(0),max_corner(1),min_corner(2)});
    vEV.push_back({max_corner(0),min_corner(1),max_corner(2)});
    vEV.push_back({max_corner(0),max_corner(1),max_corner(2)});
    vEV.push_back({min_corner(0),min_corner(1),max_corner(2)});
    vEV.push_back({min_corner(0),max_corner(1),max_corner(2)});
  }
  igl::list_to_matrix(vEV,EV);
  igl::list_to_matrix(vEE,EI);
};