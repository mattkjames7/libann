cmake_minimum_required(VERSION 3.20)

project(libann VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ann SHARED
    src/activationfunctions.cc
    src/backpropagate.cc
    src/boxcox.cc
    src/cliplog.cc
    src/costfunction.cc
    src/matrix.cc
    src/matrixarray.cc
    src/matrixmath.cc
    src/networkfunc.cc
    src/randomnumber.cc
    src/readarray.cc
    src/regularization.cc
    src/softmax.cc
)

target_include_directories(ann
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ann PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ann
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/ann.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BUILD_TESTING)
    include(FetchContent)

    set(INSTALL_GTEST OFF)
    set(BUILD_GMOCK OFF)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(ann_tests
        test/activation_tests.cc
        test/loss_regularization_tests.cc
        test/matrix_ops_tests.cc
        test/networkfunc_integration_tests.cc
        test/test_main.cc
    )

    target_include_directories(ann_tests PRIVATE test)
    target_link_libraries(ann_tests PRIVATE ann GTest::gtest)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ann_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    include(GoogleTest)
    gtest_discover_tests(
        ann_tests
        DISCOVERY_TIMEOUT 30
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
    )
endif()