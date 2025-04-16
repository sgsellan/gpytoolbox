if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 08be0704c26359bcdbe17449054e6c56b9b7538c
)
FetchContent_MakeAvailable(libigl)