#include "contouring.h"

#include <Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using EigenDStride =
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

template <typename MatrixType>
using EigenDRef =
    Eigen::Ref<MatrixType, 0, EigenDStride>;

void binding_dcsdd(py::module& m)
{
    m.def(
        "_dcsdd_cpp_impl",
        [](
            EigenDRef<const Eigen::VectorXd> S,
            EigenDRef<const Eigen::MatrixXd> GV,
            int nx,
            int ny,
            int nz,
            double isovalue,
            int outer_iters,
            int inner_iters,
            bool hermite_update,
            double mu,
            double dc_weight,
            double svd_threshold,
            double new_hermite_pos_weight,
            double new_face_pos_weight,
            double new_hermite_normal_weight,
            int batch_size,
            bool verbose
        )
        {
            ContouringOptions options;

            options.outer_iters = outer_iters;
            options.inner_iters = inner_iters;
            options.hermite_update = hermite_update;
            options.mu = mu;
            options.dc_weight = dc_weight;
            options.svd_threshold = svd_threshold;
            options.new_hermite_pos_weight =
                new_hermite_pos_weight;
            options.new_face_pos_weight =
                new_face_pos_weight;
            options.new_hermite_normal_weight =
                new_hermite_normal_weight;
            options.batch_size = batch_size;
            options.verbose = verbose;

            // This argument is not exposed, it should always be set to 1.0
            options.sphere_weight = 1.0;

            /*
             * Shift the samples so that the requested isovalue becomes
             * the zero level set.
             */
            const Eigen::VectorXd shifted_S =
                S.array() - isovalue;

            Eigen::MatrixXd V;
            Eigen::MatrixXi F;

            contouring(
                shifted_S,
                GV,
                nx,
                ny,
                nz,
                0.0,
                V,
                F,
                options
            );

            return py::make_tuple(V, F);
        },
        py::arg("S"),
        py::arg("GV"),
        py::arg("nx"),
        py::arg("ny"),
        py::arg("nz"),
        py::arg("isovalue") = 0.0,
        py::arg("outer_iters") = 100,
        py::arg("inner_iters") = 100,
        py::arg("hermite_update") = true,
        py::arg("mu") = 0.1,
        py::arg("dc_weight") = 0.02,
        py::arg("svd_threshold") = 0.01,
        py::arg("new_hermite_pos_weight") = 0.2,
        py::arg("new_face_pos_weight") = 0.2,
        py::arg("new_hermite_normal_weight") = 0.2,
        py::arg("batch_size") = 200000,
        py::arg("verbose") = false
    );
}