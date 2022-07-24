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


list(APPEND CMAKE_MODULE_PATH "${EXT_DIR}/numpyeigen/cmake")
include(numpyeigen)
option(LIBIGL_COPYLEFT_CGAL "Use CGAL"          ON)
option(LIBIGL_EMBREE "Build target igl::embree" ON)
include(libigl)

add_library(cpytoolbox
	SHARED
	# Headers
	src/upper_envelope.h
	src/ray_mesh_intersect_aabb.h
	src/in_element_aabb.h
	src/read_obj.h
	src/write_obj.h
	src/remesher/collapse_edges.h
	src/remesher/equalize_valences.h
	src/remesher/remesh_botsch.h
	src/remesher/split_edges_until_bound.h
	src/remesher/split_edges.h
	src/remesher/tangential_relaxation.h
	# Source
	src/upper_envelope.cpp
	src/ray_mesh_intersect_aabb.cpp
	src/in_element_aabb.cpp
	src/read_obj.cpp
	src/write_obj.cpp
	src/remesher/collapse_edges.cpp
	src/remesher/equalize_valences.cpp
	src/remesher/remesh_botsch.cpp
	src/remesher/split_edges_until_bound.cpp
	src/remesher/split_edges.cpp
	src/remesher/tangential_relaxation.cpp
	)

target_link_libraries(cpytoolbox igl::core igl_copyleft::cgal igl::embree)


npe_add_module(gpytoolbox_eigen_bindings
	BINDING_SOURCES
	src/binding_offset_surface.cpp
  	src/binding_booleans.cpp
	src/binding_ray_mesh_intersect.cpp
	src/binding_upper_envelope.cpp
	src/binding_in_element_aabb.cpp
	src/binding_decimate.cpp
	src/binding_remesher_botsch.cpp
	src/binding_write_obj.cpp
	src/binding_read_obj.cpp
)

target_link_libraries(gpytoolbox_eigen_bindings PUBLIC cpytoolbox igl::core igl_copyleft::cgal)

target_include_directories(gpytoolbox_eigen_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
