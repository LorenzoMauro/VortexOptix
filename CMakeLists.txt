# Vortex/CMakeLists.txt
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0076 NEW)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
cmake_minimum_required(VERSION 3.17)

#UTILS
function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

#PATHS
if(NOT CMAKE_TOOLCHAIN_FILE AND NOT UNIX)
    set(CMAKE_TOOLCHAIN_FILE "" CACHE FILEPATH "Path to toolchain file supplied to cmake.")
    message(FATAL_ERROR "No toolchain set. Please set CMAKE_TOOLCHAIN_FILE to <vcpkg_root>/scripts/buildsystems/vcpkg.cmake")
endif()

set(VCPKG_TARGET_TRIPLET x64-windows-static-md)

if(NOT CLANG_12_PATH)
    set(CLANG_12_PATH "" CACHE FILEPATH "/path/to/clang")
    message(FATAL_ERROR "No Clang 12 Selected. Please set Clang 12 folder to probably pathToInstallation/LLVM/bin/clang.exe")
endif()

if(NOT CUDA_8_PATH)
    set(CUDA_8_PATH "" CACHE PATH "/path/to/cuda")
    message(FATAL_ERROR "No Cuda 8 Slected. Please set CUDA 8 folder to C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0")
endif()

set(EXTERNAL ${CMAKE_SOURCE_DIR}/ext)

#TORCH
if(NOT TORCH_INSTALL_PREFIX_DEBUG)
    set(TORCH_INSTALL_PREFIX_DEBUG "" CACHE PATH "Path to libtorch Debug")
    message(FATAL_ERROR "Please select Torch Debug intallation Path")
endif()
if(NOT TORCH_INSTALL_PREFIX_RELEASE)
    set(TORCH_INSTALL_PREFIX_RELEASE "" CACHE PATH "Path to libtorch Debug")
    message(FATAL_ERROR "Please select Torch Debug intallation Path")
endif()

set(CMAKE_MODULE_PATH "${EXTERNAL}/CMake")
include("nvcuda_compile_module")

project(Vortex VERSION 1.0 LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit 11.7 EXACT REQUIRED)
find_package(OptiX REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(MDL_SDK REQUIRED) 
find_package(glfw3 CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

add_subdirectory(${EXTERNAL}/glad)
add_subdirectory(${EXTERNAL}/imnodes)
#add_subdirectory(${EXTERNAL}/ImGuiFileDialog)

#set spdlog_DIR to $EXTERNAL/spdlog/lib/cmake/spdlog

# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/ext/spdlog/cmake/")
add_subdirectory(${EXTERNAL}/spdlog)
get_target_property(SPDLOG_INCLUDE_DIRS spdlog::spdlog_header_only INTERFACE_INCLUDE_DIRECTORIES)

set(CU_MIN_ARCH 75)
set(CU_MAX_ARCH 86)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF)
set(ENV{TCNN_CUDA_ARCHITECTURES} "${CU_MIN_ARCH};${CU_MAX_ARCH}")
add_subdirectory(${EXTERNAL}/tcnn)
get_target_property(TCNN_INCLUDE_DIRECTORIES tiny-cuda-nn INCLUDE_DIRECTORIES)


configure_file(${CMAKE_SOURCE_DIR}/assets/VortexIco.rc.in ${CMAKE_SOURCE_DIR}/assets/VortexIco.rc @ONLY)

add_subdirectory(Vortex/src)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# Organize them into folders
set_property(TARGET Glad PROPERTY FOLDER "Dependencies")
#set_property(TARGET ImGuiFileDialog PROPERTY FOLDER "Dependencies")
set_property(TARGET imnodes PROPERTY FOLDER "Dependencies")
set_property(TARGET tiny-cuda-nn PROPERTY FOLDER "Dependencies")
set_property(TARGET fmt PROPERTY FOLDER "Dependencies")

configure_file(${CMAKE_SOURCE_DIR}/scripts/VortexInstaller.nsi.in ${CMAKE_SOURCE_DIR}/scripts/VortexInstaller.nsi @ONLY)
