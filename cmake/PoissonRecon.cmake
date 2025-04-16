if(TARGET PoissonRecon)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    PoissonRecon
    GIT_REPOSITORY https://github.com/mkazhdan/PoissonRecon.git
    GIT_TAG eaff5e54d63ba9833d1118abdacd7e35b7693f54
)
FetchContent_MakeAvailable(PoissonRecon)