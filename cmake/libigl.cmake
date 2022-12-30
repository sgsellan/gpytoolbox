if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/sgsellan/libigl.git
    GIT_TAG b54c4e30e24c7831c93c0acbbaf1cd35498d1462
)
FetchContent_MakeAvailable(libigl)
