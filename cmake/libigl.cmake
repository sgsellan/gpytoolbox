if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG ba69acc509632c56c738f03440841cc37da50497
)
FetchContent_MakeAvailable(libigl)