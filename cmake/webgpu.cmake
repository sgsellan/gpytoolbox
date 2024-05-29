if(TARGET webgpu)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    webgpu
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution.git
    GIT_TAG 9dd47f8515dfd7112b750da07e719460a88bf2e8
)
FetchContent_MakeAvailable(webgpu)