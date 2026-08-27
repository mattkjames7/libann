include(GNUInstallDirs)

add_library(ann SHARED
    activationfunctions.cc
    backpropagate.cc
    boxcox.cc
    cliplog.cc
    costfunction.cc
    matrix.cc
    matrixarray.cc
    matrixmath.cc
    networkfunc.cc
    randomnumber.cc
    readarray.cc
    regularization.cc
    softmax.cc
)

set_target_properties(ann PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# The public API does not yet use explicit __declspec(dllexport)
# annotations, so generate the DLL export table for native MSVC builds.
if(WIN32)
    set_target_properties(ann PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(ann
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ann PUBLIC OpenMP::OpenMP_CXX)

install(TARGETS ann
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/ann.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
