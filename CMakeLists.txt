cmake_minimum_required(VERSION 3.15...3.27)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(LIBRARIES)
set(TARGET_PROPERTIES)
set(COMPILE_OPTIONS)
if(NOT WIN32)
    list(APPEND COMPILE_OPTIONS -Wfatal-errors)
endif()
set(LINK_OPTIONS)
set(INCLUDE_DIRECTORIES)
set(LINK_DIRECTORIES)

find_package(pybind11 CONFIG REQUIRED)
list(APPEND LIBRARIES pybind11::headers)

set(CSRC_DIR src/ai3/csrc)
list(APPEND INCLUDE_DIRECTORIES ${CSRC_DIR})

set(CSRC_FILES
    ${CSRC_DIR}/ai3.cpp
)

set(USE_MPS_METAL NO)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_library(metal NAMES Metal)
    find_library(mps NAMES MetalPerformanceShaders)
    find_library(foundation NAMES Foundation)
    find_library(coreml NAMES CoreML)
    if(metal AND mps AND foundation AND coreml)
        message(STATUS "Found Metal, Foundation, MetalPerformanceShaders, CoreML")
        list(APPEND LIBRARIES ${metal} ${mps} ${foundation} ${coreml})
        set(USE_MPS_METAL YES)
    endif()
endif()

if(NOT "$ENV{LIBRARY_PATH}" STREQUAL "")
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
endif()

find_package(CUDAToolkit)
set(USE_CUBLAS NO)
set(USE_CUDNN NO)
if(CUDAToolkit_FOUND)
    message(WARNING "Found Toolkit")
    list(APPEND TARGET_PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED TRUE
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )
    list(APPEND LIBRARIES CUDA::cudart)

    if (TARGET CUDA::cublas)
        set(USE_CUBLAS YES)
        message(STATUS "Found cuBLAS")
        list(APPEND LIBRARIES CUDA::cublas)
    endif()

    find_library(CUDNN_LIBRARY cudnn
        PATHS ${SYSTEM_LIBRARY_PATHS}
        PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
    if(CUDNN_LIBRARY)
        get_filename_component(CUDNN_LIBRARY_DIR ${CUDNN_LIBRARY} DIRECTORY)
        find_path(CUDNN_INCLUDE_DIR cudnn.h
            PATHS ${CUDNN_LIBRARY_DIR}/../include ${CUDNN_LIBRARY_DIR}/include
            PATH_SUFFIXES include cuda/include)
        if(CUDNN_INCLUDE_DIR)
            set(USE_CUDNN YES)
            message(STATUS "Found cuDNN")
            list(APPEND LIBRARIES cudnn)
        endif()
    endif()
endif()

find_package(SYCL)
set(USE_SYCL no)
if(SYCL_FOUND)
    message(STATUS "Found SYCL")
    set(USE_SYCL YES)
    set(CMAKE_CXX_COMPILER "${SYCL_COMPILER}")

    list(APPEND INCLUDE_DIRECTORIES ${SYCL_INCLUDE_DIR} ${SYCL_SYCL_INCLUDE_DIR})
    list(APPEND LINK_DIRECTORIES ${SYCL_LIBRARY_DIR})

    separate_arguments(SYCL_CFLAGS)
    list(APPEND COMPILE_OPTIONS ${SYCL_CFLAGS})

    separate_arguments(SYCL_LFLAGS)
    list(APPEND LINK_OPTIONS ${SYCL_LFLAGS})
endif()

function(check_and_add_impl file platform_label platform_ext)
    get_filename_component(FILENAME_WE ${file} NAME_WE)
    get_filename_component(FILE_DIR ${file} DIRECTORY)

    string(REPLACE "_plain" "_${platform_label}" NEW_FILENAME "${FILENAME_WE}.${platform_ext}")
    set(NEW_FILE "${FILE_DIR}/${NEW_FILENAME}")

    if(EXISTS "${NEW_FILE}")
        list(REMOVE_ITEM IMPLS "${file}")
        list(APPEND IMPLS "${NEW_FILE}")
        set(IMPLS ${IMPLS} PARENT_SCOPE)
    endif()
endfunction()

function(use_supported_platform_impls op)
    file(GLOB IMPLS "${CSRC_DIR}/${op}/*_plain.cpp")

    if(USE_CUDNN)
        foreach(FILE ${IMPLS})
            check_and_add_impl(${FILE} "cudnn" "cpp")
        endforeach()
    endif()

    if(USE_CUBLAS)
        foreach(FILE ${IMPLS})
            check_and_add_impl(${FILE} "cublas" "cpp")
        endforeach()
    endif()

    if(USE_MPS_METAL)
        foreach(FILE ${IMPLS})
            check_and_add_impl(${FILE} "mps" "mm")
        endforeach()
        foreach(FILE ${IMPLS})
            check_and_add_impl(${FILE} "metal" "mm")
        endforeach()
    endif()

    if(USE_SYCL)
        foreach(FILE ${IMPLS})
            check_and_add_impl(${FILE} "sycl" "cpp")
        endforeach()
    endif()

    list(APPEND CSRC_FILES ${IMPLS})
    set(CSRC_FILES ${CSRC_FILES} PARENT_SCOPE)
endfunction()

use_supported_platform_impls("conv2d")
use_supported_platform_impls("linear")
use_supported_platform_impls("avgpool2d")
use_supported_platform_impls("adaptiveavgpool2d")
use_supported_platform_impls("maxpool2d")
use_supported_platform_impls("relu")
use_supported_platform_impls("flatten")

pybind11_add_module(_core MODULE ${CSRC_FILES})
set_target_properties(_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    ${TARGET_PROPERTIES}
)
target_compile_options(_core PRIVATE ${COMPILE_OPTIONS})
target_link_options(_core PRIVATE ${LINK_OPTIONS})
target_link_libraries(_core PRIVATE ${LIBRARIES})
target_include_directories(_core PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_core PRIVATE ${LINK_DIRECTORIES})

if(USE_MPS_METAL)
    target_compile_definitions(_core PRIVATE USE_MPS_METAL)
endif()

if(USE_CUBLAS)
    target_compile_definitions(_core PRIVATE USE_CUBLAS)
endif()

if(USE_CUDNN)
    target_compile_definitions(_core PRIVATE USE_CUDNN)
endif()

if(USE_SYCL)
    target_compile_definitions(_core PRIVATE USE_SYCL)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/custom.cmake")
    message(STATUS "including custom cmake")
    include("${CMAKE_SOURCE_DIR}/cmake/custom.cmake")
else()
    message(STATUS "no custom cmake found")
endif()

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
