#include <igl/embree/EmbreeIntersector.h>
#include <igl/Hit.h>
#include <igl/parallel_for.h>

void ray_mesh_intersect_aabb(const Eigen::MatrixXd& sources, const Eigen::MatrixXd& directions,const Eigen::MatrixXd& VT, const Eigen::MatrixXi& FT, Eigen::VectorXd & ts, Eigen::VectorXi & ids, Eigen::MatrixXd & lambdas)
{

ts.resize(sources.rows());
ids.resize(sources.rows());
lambdas.resize(sources.rows(),3);
igl::embree::EmbreeIntersector ei;
ei.init(VT.cast<float>(),FT,true);
//for(int si = 0;si<n;si++)
const int n = sources.rows();
igl::parallel_for(n,[&](const int si)
{
Eigen::Vector3f s = sources.row(si).cast<float>();
Eigen::Vector3f d = directions.row(si).cast<float>();
igl::Hit hit;
const float tnear = 1e-4f;
if(ei.intersectRay(s,d,hit,tnear))
{
    ids(si) = hit.id;
    ts(si) = hit.t;
    lambdas(si,0) = 1.0-hit.u-hit.v;
    lambdas(si,1) = hit.u;
    lambdas(si,2) = hit.v;
}else
{
    ids(si) = -1;
    ts(si) = std::numeric_limits<float>::infinity();
    lambdas.row(si).setZero();
}
}
);

}