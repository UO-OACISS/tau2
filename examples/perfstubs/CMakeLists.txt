cmake_minimum_required(VERSION 2.8.2)
project( perfstubs-example)

include(ExternalProject)
ExternalProject_Add(perfstubs
    GIT_REPOSITORY    https://github.com/UO-OACISS/perfstubs.git
    GIT_TAG          master
    SOURCE_DIR        "${CMAKE_BINARY_DIR}/perfstubs"
    BINARY_DIR        "${CMAKE_BINARY_DIR}/perfstubs"
    INSTALL_DIR        "${CMAKE_BINARY_DIR}/perfstubs"
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DPERFSTUBS_BUILD_EXAMPLES=TRUE ${CMAKE_BINARY_DIR}/perfstubs
    BUILD_COMMAND     "make"
    INSTALL_COMMAND   cp ${CMAKE_BINARY_DIR}/perfstubs/examples/perfstubs_test_api_c_no_tool ${CMAKE_BINARY_DIR}
    TEST_COMMAND      ""
)
