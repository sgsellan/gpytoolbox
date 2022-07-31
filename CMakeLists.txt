cmake_minimum_required(VERSION 3.10)

project(Bindings
	DESCRIPTION
		"Python libigl bindings"
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${EXT_DIR}/numpyeigen/cmake")
include(numpyeigen)
option(LIBIGL_COPYLEFT_CGAL "Use CGAL"          ON)
option(LIBIGL_EMBREE "Build target igl::embree" ON)
include(libigl)

add_library(cpytoolbox
	SHARED
	# Headers
	src/cpp/upper_envelope.h
	src/cpp/ray_mesh_intersect_aabb.h
	src/cpp/in_element_aabb.h
	src/cpp/read_obj.h
	src/cpp/write_obj.h
	src/cpp/remesher/collapse_edges.h
	src/cpp/remesher/equalize_valences.h
	src/cpp/remesher/remesh_botsch.h
	src/cpp/remesher/split_edges_until_bound.h
	src/cpp/remesher/split_edges.h
	src/cpp/remesher/tangential_relaxation.h
	# Source
	src/cpp/upper_envelope.cpp
	src/cpp/ray_mesh_intersect_aabb.cpp
	src/cpp/in_element_aabb.cpp
	src/cpp/read_obj.cpp
	src/cpp/write_obj.cpp
	src/cpp/remesher/collapse_edges.cpp
	src/cpp/remesher/equalize_valences.cpp
	src/cpp/remesher/remesh_botsch.cpp
	src/cpp/remesher/split_edges_until_bound.cpp
	src/cpp/remesher/split_edges.cpp
	src/cpp/remesher/tangential_relaxation.cpp
	)

# target_link_libraries(cpytoolbox igl::core igl_copyleft::cgal igl::embree)
target_link_libraries(cpytoolbox igl::core igl::embree igl_copyleft::cgal)


npe_add_module(gpytoolbox_bindings
	BINDING_SOURCES
	src/cpp/binding_offset_surface.cpp
  	src/cpp/binding_booleans.cpp
	src/cpp/binding_ray_mesh_intersect.cpp
	src/cpp/binding_upper_envelope.cpp
	src/cpp/binding_in_element_aabb.cpp
	src/cpp/binding_decimate.cpp
	src/cpp/binding_remesher_botsch.cpp
	src/cpp/binding_write_obj.cpp
	src/cpp/binding_read_obj.cpp
)

# target_link_libraries(gpytoolbox_bindings PUBLIC cpytoolbox igl::core igl_copyleft::cgal)
target_link_libraries(gpytoolbox_bindings PUBLIC cpytoolbox igl::core)

target_include_directories(gpytoolbox_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")