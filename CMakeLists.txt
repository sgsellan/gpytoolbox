cmake_minimum_required(VERSION 3.10)

project(Bindings
	DESCRIPTION
		"Python libigl bindings"
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(EXT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


list(APPEND CMAKE_MODULE_PATH ${EXT_DIR}/numpyeigen/cmake)
include(numpyeigen)
option(LIBIGL_COPYLEFT_CGAL "Use CGAL"          ON)
include(libigl)

add_library(cpytoolbox
	SHARED
	# Headers
	src/upper_envelope.h
	# Source
	src/upper_envelope.cpp
	)

target_link_libraries(cpytoolbox igl::core igl_copyleft::cgal)


npe_add_module(gpytoolbox_eigen_bindings
	BINDING_SOURCES
	src/bindings.cpp
)

target_link_libraries(gpytoolbox_eigen_bindings PUBLIC cpytoolbox igl::core igl_copyleft::cgal)

target_include_directories(gpytoolbox_eigen_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
