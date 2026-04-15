include(FetchContent)
find_package(OpenMP)

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
    activation_tests.cc
    loss_regularization_tests.cc
    matrix_ops_tests.cc
    networkfunc_integration_tests.cc
    test_main.cc
)

set_target_properties(ann_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_include_directories(ann_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ann_tests PRIVATE ann GTest::gtest)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ann_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)
gtest_discover_tests(
    ann_tests
    DISCOVERY_TIMEOUT 30
    WORKING_DIRECTORY $<TARGET_FILE_DIR:ann_tests>
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1"
)
