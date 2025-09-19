cmake_minimum_required(VERSION 3.26)

# https://discourse.cmake.org/t/msvc-runtime-library-completely-ignored/10004
cmake_policy(SET CMP0091 NEW)

project(Bindings
	DESCRIPTION
		"Python bindings"
)

# MSVC needs explicit configuration for multithreading
# Select a multi-threaded statically-linked runtime library
# 	with or without debug information depending on the configuration
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "15.0.0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ld_classic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ld_classic")
  endif()
endif()


# Include libigl
option(LIBIGL_COPYLEFT_CGAL "Use CGAL"          ON)
option(LIBIGL_EMBREE "Build target igl::embree" ON)
set(BOOST_ROOT "/Users/silviasellan/Dropbox/temp/libigl/build/_deps/boost-src")
include(libigl)
igl_include(glfw)
igl_include(embree)

# Include PoissonRecon
include(PoissonRecon)
add_library(PoissonRecon INTERFACE)
target_include_directories(PoissonRecon INTERFACE "${CMAKE_BINARY_DIR}/_deps/poissonrecon-src/Src/")

# Include nanoflann
include(nanoflann)

# Include webgpu
include(webgpu)
include(glfw3webgpu)
if(TARGET webgpu AND TARGET glfw3webgpu)
	add_compile_definitions(GL_AVAILABLE)
	set(GL_LIBS webgpu glfw3webgpu)
else()
	set(GL_LIBS "")
endif()

# List of all libraries to link
set(LIBRARIES_TO_LINK igl::core igl::embree nanoflann PoissonRecon ${GL_LIBS})
set(COPYLEFT_LIBRARIES_TO_LINK igl::core igl_copyleft::cgal)

add_library(cpytoolbox
	STATIC
	# SHARED
	# Headers
	src/cpp/upper_envelope.h
	src/cpp/ray_mesh_intersect_aabb.h
	src/cpp/in_element_aabb.h
	src/cpp/read_obj.h
	src/cpp/write_obj.h
	src/cpp/point_cloud_to_mesh.h
	src/cpp/reach_for_the_arcs/outside_points_from_rasterization.h
	src/cpp/reach_for_the_arcs/locally_make_feasible.h
	src/cpp/reach_for_the_arcs/fine_tune_point_cloud_iter.h
	src/cpp/reach_for_the_arcs/sAABB.h
	src/cpp/reach_for_the_arcs/resolve_collisions_on_sphere.h
	src/cpp/remesher/collapse_edges.h
	src/cpp/remesher/equalize_valences.h
	src/cpp/remesher/remesh_botsch.h
	src/cpp/remesher/split_edges_until_bound.h
	src/cpp/remesher/split_edges.h
	src/cpp/remesher/tangential_relaxation.h
	src/cpp/microstl/microstl_wrappers.h
	src/cpp/tinyply/tinyply_wrappers.h
	# Source
	src/cpp/upper_envelope.cpp
	src/cpp/ray_mesh_intersect_aabb.cpp
	src/cpp/in_element_aabb.cpp
	src/cpp/read_obj.cpp
	src/cpp/write_obj.cpp
	src/cpp/point_cloud_to_mesh.cpp
	src/cpp/reach_for_the_arcs/outside_points_from_rasterization.cpp
	src/cpp/reach_for_the_arcs/locally_make_feasible.cpp
	src/cpp/reach_for_the_arcs/fine_tune_point_cloud_iter.cpp
	src/cpp/reach_for_the_arcs/sAABB.cpp
	src/cpp/reach_for_the_arcs/resolve_collisions_on_sphere.cpp
	src/cpp/remesher/collapse_edges.cpp
	src/cpp/remesher/equalize_valences.cpp
	src/cpp/remesher/remesh_botsch.cpp
	src/cpp/remesher/split_edges_until_bound.cpp
	src/cpp/remesher/split_edges.cpp
	src/cpp/remesher/tangential_relaxation.cpp
	src/cpp/microstl/microstl_wrappers.cpp
	src/cpp/tinyply/tinyply_wrappers.cpp
	# microstl
	src/cpp/microstl/microstl.h
	# tinyply
	src/cpp/tinyply/tinyply.h
	)

	add_library(cpytoolbox_copyleft
	STATIC
	# SHARED
	# Headers
	src/cpp/swept_volume/fd_interpolate.cpp
	src/cpp/swept_volume/gradient_descent_test.cpp
	src/cpp/swept_volume/random_points_on_mesh.cpp
	src/cpp/swept_volume/sparse_continuation.cpp
	src/cpp/swept_volume/swept_volume.cpp
	# Source
	src/cpp/swept_volume/fd_interpolate.h
	src/cpp/swept_volume/gradient_descent_test.h
	src/cpp/swept_volume/random_points_on_mesh.h
	src/cpp/swept_volume/sparse_continuation.h
	src/cpp/swept_volume/swept_volume.h
	)

target_link_libraries(cpytoolbox ${LIBRARIES_TO_LINK})
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "17.0")
    #To make PSR work on newer Clangs
    target_compile_options(cpytoolbox PRIVATE -Wno-missing-template-arg-list-after-template-kw)
  endif()
endif()
set(LIBRARIES_TO_LINK cpytoolbox ${LIBRARIES_TO_LINK})
target_link_libraries(cpytoolbox_copyleft ${COPYLEFT_LIBRARIES_TO_LINK})
set(COPYLEFT_LIBRARIES_TO_LINK cpytoolbox_copyleft ${COPYLEFT_LIBRARIES_TO_LINK})

# Otman's python bindings using pybind11
add_subdirectory(./ext/pybind11/)
pybind11_add_module(gpytoolbox_bindings
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/gpytoolbox_bindings_core.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_read_obj.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_write_obj.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_decimate.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_fast_winding_number.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_hausdorff_distance.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_in_element_aabb.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_marching_cubes.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_offset_surface.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_point_mesh_squared_distance.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_ray_mesh_intersect.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_read_stl.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_write_stl.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_remesher_botsch.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_upper_envelope.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_read_ply.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_write_ply.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_curved_hessian_intrinsic.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_point_cloud_to_mesh.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/reach_for_the_arcs/binding_outside_points_from_rasterization.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/reach_for_the_arcs//binding_locally_make_feasible.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/reach_for_the_arcs//binding_fine_tune_point_cloud_iter.cpp"
)

pybind11_add_module(gpytoolbox_bindings_copyleft
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/gpytoolbox_bindings_copyleft_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_swept_volume.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_booleans.cpp"
)

target_link_libraries(gpytoolbox_bindings PUBLIC ${LIBRARIES_TO_LINK})
target_link_libraries(gpytoolbox_bindings_copyleft PUBLIC ${COPYLEFT_LIBRARIES_TO_LINK})

target_include_directories(gpytoolbox_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/")
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	add_custom_command(TARGET gpytoolbox_bindings POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpytoolbox_bindings> $<TARGET_RUNTIME_DLLS:gpytoolbox_bindings> $<TARGET_FILE_DIR:gpytoolbox_bindings>
		COMMAND_EXPAND_LISTS)
endif()

target_include_directories(gpytoolbox_bindings_copyleft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	add_custom_command(TARGET gpytoolbox_bindings_copyleft POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpytoolbox_bindings_copyleft> $<TARGET_RUNTIME_DLLS:gpytoolbox_bindings_copyleft> $<TARGET_FILE_DIR:gpytoolbox_bindings_copyleft>
		COMMAND_EXPAND_LISTS)
endif()

# Manually copy wgpu to the right place
target_copy_webgpu_binaries(gpytoolbox_bindings)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	# Extra difficult on mac.
	add_custom_command(
		TARGET gpytoolbox_bindings POST_BUILD
		COMMAND
			${CMAKE_INSTALL_NAME_TOOL} "-change"
			"@executable_path/libwgpu_native.dylib"
			"$<TARGET_FILE_DIR:gpytoolbox_bindings>/libwgpu_native.dylib"
			"$<TARGET_FILE:gpytoolbox_bindings>"
		VERBATIM
	)
endif()

