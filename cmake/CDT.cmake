if(TARGET CDT::CDT)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    CDT
    GIT_REPOSITORY https://github.com/artem-ogre/CDT.git
    GIT_TAG 3765e08ec4bbef0c6b721204d7c0e45d279084e9
    SOURCE_SUBDIR CDT
)
FetchContent_MakeAvailable(CDT)