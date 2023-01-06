project(TauProfiling)

cmake_minimum_required(VERSION 3.14)
set (CMAKE_CXX_STANDARD 14)

#set( LLVM_ROOT "" CACHE PATH "Root of LLVM install." )
# A bit of a sanity check :
#if( NOT EXISTS ${LLVM_ROOT}/include/llvm )
#    message( FATAL_ERROR "LLVM_ROOT (${LLVM_ROOT}) is invalid" )
#endif ()

# Load LLVM config
list( APPEND CMAKE_PREFIX_PATH "${LLVM_DIR}/share/llvm/cmake" )
find_package( LLVM REQUIRED CONFIG )

list( APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" )
include( HandleLLVMOptions ) 
include( AddLLVM )

add_definitions(${LLVM_DEFINITIONS} )
include_directories(${LLVM_INCLUDE_DIRS} )

if(NOT DEFINED CLANG_VERSION_MAJOR)
  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR} )
endif()

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

# LLVM 13 is using a new pass manager. Use it, unless asked. 
OPTION(LEGACY_PM "Use the legacy pass manager with LLVM 13" OFF) 
if(${CLANG_VERSION_MAJOR} VERSION_GREATER_EQUAL 14)
  # C++14 is not enough anymore
    set (CMAKE_CXX_STANDARD 17)
    ADD_DEFINITIONS(-DLEGACY_PM=0)
else()
  ADD_DEFINITIONS(-DLEGACY_PM=1)
endif()
  
# Propagate to our plugin
add_definitions ( ${LLVM_DEFINITIONS} )
include_directories ( ${LLVM_INCLUDE_DIRS} )
set( LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib/${CMAKE_CFG_INT_DIR}" )
set( LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib/${CMAKE_CFG_INT_DIR}" )

# required for `add_llvm_loadable_module`
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" )
include(AddLLVM)

add_subdirectory( src )

