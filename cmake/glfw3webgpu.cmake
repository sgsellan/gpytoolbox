if(TARGET glfw3webgpu)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    glfw3webgpu
    GIT_REPOSITORY https://github.com/eliemichel/glfw3webgpu.git
    GIT_TAG 22e54ff192d9af967484ccf58570fe06fde1e969
)
FetchContent_MakeAvailable(glfw3webgpu)