if(TARGET CDT)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    CDT
    GIT_REPOSITORY https://github.com/artem-ogre/CDT.git
    GIT_TAG 24ff7d7969dded0243c9e1e992a5398a6d0293dd
)
FetchContent_MakeAvailable(CDT)
