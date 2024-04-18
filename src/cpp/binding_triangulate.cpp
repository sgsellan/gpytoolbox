#include <CDT.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_triangulate(py::module& m) {
    m.def("_triangulate_cpp_impl",[](EigenDRef<MatrixXd> V,
                         EigenDRef<MatrixXi> E,
                         double max_area,
                         double min_angle,
                         int max_steiner_points)
        {

            std::vector<CDT::V2d<double> > vertices;
            for(int i=0; i<V.rows(); ++i) {
                vertices.push_back(CDT::V2d<double>::make(V(i,0), V(i,1)));
            }
            CDT::Triangulation<double> cdt;
            cdt.insertVertices(vertices);
            std::vector<CDT::Edge> edges;
            if(E.size()>0) {
                for(int i=0; i<E.rows(); ++i) {
                    edges.emplace_back(E(i,0), E(i,1));
                }
                cdt.conformToEdges(edges);
            }
            if(E.size()>0) {
                cdt.eraseOuterTrianglesAndHoles();
            } else {
                cdt.eraseSuperTriangle();
            }
            Eigen::MatrixXd W(cdt.vertices.size(), 2);
            for(int i=0; i<cdt.vertices.size(); ++i) {
                W.row(i) << cdt.vertices[i].x, cdt.vertices[i].y;
            }
            Eigen::MatrixXi F(cdt.triangles.size(), 3);
            for(int i=0; i<cdt.triangles.size(); ++i) {
                for(int j=0; j<3; ++j) {
                    F(i,j) = cdt.triangles[i].vertices[j];
                }
            }

            return std::make_tuple(W, F);
        });
    
}
