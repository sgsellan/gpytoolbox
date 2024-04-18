#include "point_cloud_to_mesh.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
#include <iostream>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

template<typename Real, typename Int>
void binding_definer_point_cloud_to_mesh(py::module& m, const char *name)
{
    using Mat = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using MatI = Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>;

    m.def(name,[](EigenDRef<Mat> _cloud_points,
                     EigenDRef<Mat> _cloud_normals,
                     Real screening_weight,
                     int depth,
                     const std::string& _outerBoundaryType,
                     bool parallel,
                     bool verbose)
        {
            Mat cloud_points(_cloud_points), cloud_normals(_cloud_normals);
            Mat V;
            MatI F;
            const int dim = cloud_points.cols();
            if(dim==2) {
                if(_outerBoundaryType=="Dirichlet" ||
                _outerBoundaryType=="dirichlet") {
                    point_cloud_to_mesh<Real,Int,
                        PointCloudReconstructionOuterBoundaryType::Dirichlet,2>
                        (cloud_points, cloud_normals,
                        screening_weight,
                        depth,
                        parallel, verbose,
                        V, F);
                } else { //Neumann
                    point_cloud_to_mesh<Real,Int,
                    PointCloudReconstructionOuterBoundaryType::Neumann,2>
                    (cloud_points, cloud_normals,
                    screening_weight,
                    depth,
                    parallel, verbose,
                    V, F);
                }
            } else if(dim==3) {
                if(_outerBoundaryType=="Dirichlet" ||
                _outerBoundaryType=="dirichlet") {
                    point_cloud_to_mesh<Real,Int,
                        PointCloudReconstructionOuterBoundaryType::Dirichlet,3>
                        (cloud_points, cloud_normals,
                        screening_weight,
                        depth,
                        parallel, verbose,
                        V, F);
                } else { //Neumann
                    point_cloud_to_mesh<Real,Int,
                    PointCloudReconstructionOuterBoundaryType::Neumann,3>
                    (cloud_points, cloud_normals,
                    screening_weight,
                    depth,
                    parallel, verbose,
                    V, F);
                }
            }
            return std::make_tuple(V, F);
        });
}

void binding_point_cloud_to_mesh(py::module& m) {
    binding_definer_point_cloud_to_mesh<double, int>(m, "_point_cloud_to_mesh_cpp_impl");
}

