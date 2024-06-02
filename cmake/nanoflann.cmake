if(TARGET nanoflann)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG e0a985204e999a2caed0fdbb71ee8c556621a4f3
)
FetchContent_MakeAvailable(nanoflann)