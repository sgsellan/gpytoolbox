
#include <igl/opengl/glfw/Viewer.h>

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


namespace iglv = igl::opengl::glfw;

// Pybind does not like libigl's default viewer constructor.
//Instead of changing the igl one, just wrap it. 


void data_list_check(iglv::Viewer& v, int id)
{
    if (v.data_list.size() >= id) {
        printf("Data list entry not initialized, please \n \
                    add mesh with gpytoolbox.append_mesh(V, F), \n  \
                    or ensure data list has the correct size \n"); 
        //TODO: expose the data list as something a 
        //viewer can easily browse through
    }
}

//TODO: make it known to the user what types of colormaps exist
igl::ColorMapType string_to_colormap_type(std::string str)
{
    if (str == "viridis")
        return igl::COLOR_MAP_TYPE_VIRIDIS;
    else if (str == "inferno")
        return igl::COLOR_MAP_TYPE_INFERNO;
    else if (str == "magma")
        return igl::COLOR_MAP_TYPE_MAGMA;
    else if (str == "parula")
        return igl::COLOR_MAP_TYPE_PARULA;
    else if (str == "plasma")
        return igl::COLOR_MAP_TYPE_PLASMA;
    else if (str=="turbo")
        return  igl::COLOR_MAP_TYPE_TURBO;
    else if (str =="jet")
        return  igl::COLOR_MAP_TYPE_JET;
    
}

void bind_viewer(py::module& m) {
	py::class_<iglv::Viewer>(m, "viewer")
		.def(py::init<>())
        // setting mesh
        .def("set_mesh", [&](iglv::Viewer& v, EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F, int i) {
            data_list_check(v, i);
            v.data_list[i].set_mesh(V, F);
       	}, py::arg("V"), py::arg("F"), py::arg("id") = 0)

        .def("append_mesh", [&](iglv::Viewer& v, EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F) {
            v.append_mesh();
            v.data_list[v.data_list.size() - 1].set_mesh(V, F);
            return v.data_list.size() - 1;
        })

        .def("append_mesh", [&](iglv::Viewer& v) {
            v.append_mesh();
            return v.data_list.size() - 1;
        })

        //color manipulation
        .def("background_color", [&](iglv::Viewer& v, Eigen::Vector4d & color) {
            v.core().background_color = color.cast<float>(); 
            })
        

        .def("set_colors", [&](iglv::Viewer& v, Eigen::RowVector3d& color, int id)
        {
            data_list_check(v, id);
            v.data_list[id].set_colors(color);
        }, py::arg("color"), py::arg("id")=0)


        .def("set_data", [&](iglv::Viewer& v, Eigen::VectorXd& d, int id,
            std::string colormap, int num_steps)
        {
            data_list_check(v, id);
            igl::ColorMapType cmap = string_to_colormap_type(colormap);
            v.data_list[id].set_data(d, cmap, num_steps);
        }, py::arg("d"), py::arg("id") = 0, 
            py::arg("colormap") = "viridis", py::arg("num_steps")=21)

        .def("set_data", [&](iglv::Viewer& v, Eigen::VectorXd& d, double caxis_min, 
            double caxis_max, int id,  std::string colormap, int num_steps)
        {
            data_list_check(v, id);
            igl::ColorMapType cmap = string_to_colormap_type(colormap);
            v.data_list[id].set_data(d, caxis_min, caxis_max, cmap, num_steps );
        }, py::arg("d"), py::arg("caxis_min"), py::arg("caxis_max"), py::arg("id") = 0, 
            py::arg("colormap") = "viridis", py::arg("num_steps") = 21)

        //miscallaneous
        .def("show_lines", [&](iglv::Viewer& v, bool show_lines, int id){
            data_list_check(v, id); 
            v.data_list[id].show_lines = show_lines;
            }, py::arg("show_lines"), py::arg("id")=0)

        .def("show_faces", [&](iglv::Viewer& v, bool show_faces, int id) {
            data_list_check(v, id);
            v.data_list[id].show_faces = show_faces;
            }, py::arg("show_faces"), py::arg("id") = 0)

        .def("double_sided", [&](iglv::Viewer& v, bool double_sided, int id) {
            data_list_check(v, id);
            v.data_list[id].double_sided = double_sided;
            }, py::arg("double_sided"), py::arg("id") = 0)

       .def("invert_normals", [&](iglv::Viewer& v, bool invert_normals, int id) {
            data_list_check(v, id);
            v.data_list[id].invert_normals = invert_normals;
            }, py::arg("invert_normals"), py::arg("id") = 0)

       .def("is_visible", [&](iglv::Viewer& v, bool is_visible, int id) {
            data_list_check(v, id);
            v.data_list[id].is_visible = is_visible;
            }, py::arg("is_visible"), py::arg("id") = 0)

       .def("show_faces", [&](iglv::Viewer& v, bool show_faces, int id) {
            data_list_check(v, id);
            v.data_list[id].show_faces = show_faces;
            }, py::arg("show_faces"), py::arg("id") = 0)

       .def("launch", [&](iglv::Viewer& v) 
       	    {v.launch(); })
		;
}