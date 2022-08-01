#include <Eigen/Core>
#include <Eigen/Sparse>
#include <igl/AABB.h>
#include <igl/in_element.h>


void in_element_aabb(const Eigen::MatrixXd& queries, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXi & I)
{
    std::vector<Eigen::Triplet<double> > ijv;
    Eigen::MatrixXd bb_mins;
    Eigen::MatrixXd bb_maxs;
    Eigen::VectorXi elements;
    int dim = V.cols();
    bb_mins.resize(0,dim);
    bb_maxs.resize(0,dim);
    elements.resize(0,1);
    switch(V.cols())
  {
    case 3:
    {
      igl::AABB<Eigen::MatrixXd,3> aabb;
      aabb.init(V,F,bb_mins,bb_maxs,elements);
      igl::in_element(V,F,queries,aabb,I);
      break;
    }
    case 2:
    {
      igl::AABB<Eigen::MatrixXd,2> aabb;
      aabb.init(V,F,bb_mins,bb_maxs,elements);
      igl::in_element(V,F,queries,aabb,I);
      break;
    }
  }
}