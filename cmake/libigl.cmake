if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/odedstein/libigl.git
    GIT_TAG d1a3cd0e8a94c010e452ac0e96dd009200026982
)
FetchContent_MakeAvailable(libigl)