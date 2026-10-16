if(TARGET igl::core)
    return()
endif()

include(FetchContent)

# libigl clones Eigen from gitlab.com over git, which often fails under load
# ("GitLab is currently unable to handle this request due to load") and breaks
# CI. Declaring eigen first makes libigl's recipe use this release tarball of
# the same 3.4.0 tag instead (the first FetchContent_Declare of a name wins).
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
)

FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 08be0704c26359bcdbe17449054e6c56b9b7538c
)
FetchContent_MakeAvailable(libigl)