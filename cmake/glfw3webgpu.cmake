if(TARGET glfw3webgpu)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    glfw3webgpu
    GIT_REPOSITORY https://github.com/eliemichel/glfw3webgpu.git
    GIT_TAG e4cd9131c7472b000641a104116c2fecf13d55a7
)
FetchContent_MakeAvailable(glfw3webgpu)