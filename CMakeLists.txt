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

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	# libigl
	${EXT_DIR}/libigl/include
	)

list(APPEND CMAKE_MODULE_PATH ${EXT_DIR}/numpyeigen/cmake)
include(numpyeigen)

# add_library(remesh
# 	SHARED

# 	# Headers
# 	src/collapse_edges.h
# 	src/equalize_valences.h
# 	src/remesh_botsch.h
# 	src/split_edges.h
# 	src/split_edges_until_bound.h
# 	src/tangential_relaxation.h

# 	# Source
# 	src/collapse_edges.cpp
# 	src/equalize_valences.cpp
# 	src/remesh_botsch.cpp
# 	src/split_edges.cpp
# 	src/split_edges_until_bound.cpp
# 	src/tangential_relaxation.cpp

# 	)


#target_link_libraries(remesh Threads::Threads Eigen3::Eigen)

#add_executable(remeshmesh remeshmesh.cpp)
#target_link_libraries(remeshmesh remesh)

npe_add_module(pyboolean
	BINDING_SOURCES
	src/bindings.cpp
)

target_include_directories(pyboolean PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
