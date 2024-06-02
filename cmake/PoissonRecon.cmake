if(TARGET PoissonRecon)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    PoissonRecon
    GIT_REPOSITORY https://github.com/mkazhdan/PoissonRecon.git
    GIT_TAG e04a91d40093dd80669afb07f7d3f586db063ee9
)
FetchContent_MakeAvailable(PoissonRecon)