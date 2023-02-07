if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 0c77359c89f3cd7ae5cdee09e7175953fe41a3c8
)
FetchContent_MakeAvailable(libigl)