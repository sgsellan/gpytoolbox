if(TARGET CDT::CDT)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    CDT
    GIT_REPOSITORY https://github.com/artem-ogre/CDT.git
    GIT_TAG 2068d015b9db3c92481e869b0c1f669b96a1d70a
    SOURCE_SUBDIR CDT
)
FetchContent_MakeAvailable(CDT)