#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

// using namespace Eigen;
namespace py = pybind11;
// using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
// template <typename MatrixType>
// using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

//forward declare all bindings
void binding_read_obj(py::module& m);
void binding_write_obj(py::module& m);
void binding_decimate(py::module& m);
void binding_fast_winding_number(py::module& m);
void binding_hausdorff_distance(py::module& m);
void binding_in_element_aabb(py::module& m);
void binding_marching_cubes(py::module& m);
void binding_offset_surface(py::module& m);
void binding_point_mesh_squared_distance(py::module& m);
void binding_ray_mesh_intersect(py::module& m);
void binding_read_stl(py::module& m);
void binding_write_stl(py::module& m);
void binding_remesh_botsch(py::module& m);
void binding_upper_envelope(py::module& m);
void binding_read_ply(py::module& m);
void binding_write_ply(py::module& m);
void binding_per_face_prin_curvature(py::module& m);

PYBIND11_MODULE(gpytoolbox_bindings, m) {

    /// call all bindings declared above  
    binding_read_obj(m);
    binding_write_obj(m);
    binding_decimate(m);
    binding_fast_winding_number(m);
    binding_hausdorff_distance(m);
    binding_in_element_aabb(m);
    binding_marching_cubes(m);
    binding_offset_surface(m);
    binding_point_mesh_squared_distance(m);
    binding_ray_mesh_intersect(m);
    binding_read_stl(m);
    binding_write_stl(m);
    binding_remesh_botsch(m);
    binding_upper_envelope(m);
    binding_read_ply(m);
    binding_write_ply(m);
    binding_per_face_prin_curvature(m);

    m.def("help", [&]() {printf("hi"); });
}

