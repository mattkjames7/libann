include(GNUInstallDirs)

find_package(OpenMP)

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

target_include_directories(ann
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ann PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ann
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/ann.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
