#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
#include <igl/copyleft/tetgen/tetrahedralize.h>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_tetrahedralize(py::module& m) {
    m.def("_tetrahedralize_cpp_impl",[](EigenDRef<MatrixXd> _V,
                         EigenDRef<MatrixXi> _F,
                         EigenDRef<MatrixXd> _H,
                         double max_volume,
                         double min_rad_edge_ratio)
        {

            std::ostringstream params_stream;
            params_stream << "Q";
            if(max_volume>0) {
                params_stream << "a" << max_volume;
            }
            if(min_rad_edge_ratio>0) {
                params_stream << "q" << min_rad_edge_ratio;
            }
            Eigen::MatrixXd V(_V), H(_H);
            Eigen::MatrixXi F(_F);
            Eigen::MatrixXd R,W;
            Eigen::MatrixXi TR,PT;
            Eigen::MatrixXi T,TF,FT,TN;
            size_t numRegions;
            int status = igl::copyleft::tetgen::tetrahedralize(V,F,H,
                R,
                params_stream.str(),
                W,T,TF,TR,TN,PT,FT,numRegions);
            return std::make_tuple(status, W, T, TF);
        });
    
}
