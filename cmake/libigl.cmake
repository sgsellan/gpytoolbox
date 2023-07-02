if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 282388c68a0e1de325185f7866c296b3ed10a7f5
)
FetchContent_MakeAvailable(libigl)