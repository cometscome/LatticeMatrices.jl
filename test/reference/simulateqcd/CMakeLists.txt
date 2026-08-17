cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(LatticeMatricesHISQReference LANGUAGES CXX CUDA)

find_package(MPI 3.1 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(SIMULATEQCD_SOURCE_DIR "" CACHE PATH
    "Path to an unmodified SIMULATeQCD source tree")
if(NOT EXISTS "${SIMULATEQCD_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Set SIMULATEQCD_SOURCE_DIR to the SIMULATeQCD source tree")
endif()

# Reuse SIMULATeQCD's normal configuration and its CodeBase object library,
# while keeping this reference program outside the SIMULATeQCD worktree.
add_subdirectory("${SIMULATEQCD_SOURCE_DIR}" simulateqcd EXCLUDE_FROM_ALL)

set(REFERENCE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/hisq_level1_reference.cpp")
set(HISQ_SMEARING_SOURCE
    "${SIMULATEQCD_SOURCE_DIR}/src/modules/hisq/hisqSmearing.cpp")
set_source_files_properties(
    "${REFERENCE_SOURCE}" "${HISQ_SMEARING_SOURCE}"
    PROPERTIES LANGUAGE CUDA)

add_executable(simulateqcd_hisq_level1_reference
    "${REFERENCE_SOURCE}"
    "${HISQ_SMEARING_SOURCE}")
target_link_libraries(simulateqcd_hisq_level1_reference PRIVATE
    CodeBase MPI::MPI_CXX CUDA::cudart)
target_include_directories(simulateqcd_hisq_level1_reference PRIVATE
    "${SIMULATEQCD_SOURCE_DIR}/src"
    "${MPI_INCLUDE_PATH}"
    "${MPI_CXX_INCLUDE_DIRS}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_compile_definitions(simulateqcd_hisq_level1_reference PRIVATE
    USE_CUDA
    ARCHITECTURE=${ARCHITECTURE}
    GIT_HASH="external-reference"
    HALODEPTH_0=1
    DOUBLEPREC=1
    COMP_R18=1)
if(USE_GPU_AWARE_MPI)
    target_compile_definitions(
        simulateqcd_hisq_level1_reference PRIVATE USE_GPU_AWARE_MPI)
endif()
if(USE_GPU_P2P)
    target_compile_definitions(
        simulateqcd_hisq_level1_reference PRIVATE USE_GPU_P2P)
endif()
if(COMMS_STREAMS)
    target_compile_definitions(
        simulateqcd_hisq_level1_reference PRIVATE COMMS_STREAMS)
endif()
if(USE_TILED_MULTIRHS)
    target_compile_definitions(
        simulateqcd_hisq_level1_reference PRIVATE USE_TILED_MULTIRHS)
endif()
set_target_properties(simulateqcd_hisq_level1_reference PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/reference")

set(FULL_REFERENCE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/hisq_full_reference.cpp")
set_source_files_properties(
    "${FULL_REFERENCE_SOURCE}" PROPERTIES LANGUAGE CUDA)
add_executable(simulateqcd_hisq_full_reference
    "${FULL_REFERENCE_SOURCE}"
    "${HISQ_SMEARING_SOURCE}")
target_link_libraries(simulateqcd_hisq_full_reference PRIVATE
    CodeBase MPI::MPI_CXX CUDA::cudart)
target_include_directories(simulateqcd_hisq_full_reference PRIVATE
    "${SIMULATEQCD_SOURCE_DIR}/src"
    "${MPI_INCLUDE_PATH}"
    "${MPI_CXX_INCLUDE_DIRS}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_compile_definitions(simulateqcd_hisq_full_reference PRIVATE
    USE_CUDA
    ARCHITECTURE=${ARCHITECTURE}
    GIT_HASH="external-reference"
    HALODEPTH_0=1
    DOUBLEPREC=1
    COMP_R18=1)
if(USE_GPU_AWARE_MPI)
    target_compile_definitions(
        simulateqcd_hisq_full_reference PRIVATE USE_GPU_AWARE_MPI)
endif()
if(USE_GPU_P2P)
    target_compile_definitions(
        simulateqcd_hisq_full_reference PRIVATE USE_GPU_P2P)
endif()
if(COMMS_STREAMS)
    target_compile_definitions(
        simulateqcd_hisq_full_reference PRIVATE COMMS_STREAMS)
endif()
if(USE_TILED_MULTIRHS)
    target_compile_definitions(
        simulateqcd_hisq_full_reference PRIVATE USE_TILED_MULTIRHS)
endif()
set_target_properties(simulateqcd_hisq_full_reference PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/reference")

set(BENCHMARK_SOURCE "${CMAKE_CURRENT_LIST_DIR}/hisq_benchmark.cpp")
set_source_files_properties("${BENCHMARK_SOURCE}" PROPERTIES LANGUAGE CUDA)
add_executable(simulateqcd_hisq_benchmark
    "${BENCHMARK_SOURCE}"
    "${HISQ_SMEARING_SOURCE}")
target_link_libraries(simulateqcd_hisq_benchmark PRIVATE
    CodeBase MPI::MPI_CXX CUDA::cudart)
target_include_directories(simulateqcd_hisq_benchmark PRIVATE
    "${SIMULATEQCD_SOURCE_DIR}/src"
    "${MPI_INCLUDE_PATH}"
    "${MPI_CXX_INCLUDE_DIRS}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_compile_definitions(simulateqcd_hisq_benchmark PRIVATE
    USE_CUDA
    ARCHITECTURE=${ARCHITECTURE}
    GIT_HASH="external-benchmark"
    HALODEPTH_2=1
    DOUBLEPREC=1
    COMP_R18=1)
if(USE_GPU_AWARE_MPI)
    target_compile_definitions(
        simulateqcd_hisq_benchmark PRIVATE USE_GPU_AWARE_MPI)
endif()
if(USE_GPU_P2P)
    target_compile_definitions(
        simulateqcd_hisq_benchmark PRIVATE USE_GPU_P2P)
endif()
if(COMMS_STREAMS)
    target_compile_definitions(
        simulateqcd_hisq_benchmark PRIVATE COMMS_STREAMS)
endif()
if(USE_TILED_MULTIRHS)
    target_compile_definitions(
        simulateqcd_hisq_benchmark PRIVATE USE_TILED_MULTIRHS)
endif()
set_target_properties(simulateqcd_hisq_benchmark PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/reference")

set(DIRAC_BENCHMARK_SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/hisq_dirac_benchmark.cpp")
set(SPINOR_SOURCE "${SIMULATEQCD_SOURCE_DIR}/src/spinor/spinorfield.cpp")
set_source_files_properties(
    "${DIRAC_BENCHMARK_SOURCE}" "${SPINOR_SOURCE}"
    PROPERTIES LANGUAGE CUDA)
add_executable(simulateqcd_hisq_dirac_benchmark
    "${DIRAC_BENCHMARK_SOURCE}"
    "${HISQ_SMEARING_SOURCE}"
    "${SPINOR_SOURCE}")
target_link_libraries(simulateqcd_hisq_dirac_benchmark PRIVATE
    CodeBase MPI::MPI_CXX CUDA::cudart)
target_include_directories(simulateqcd_hisq_dirac_benchmark PRIVATE
    "${SIMULATEQCD_SOURCE_DIR}/src"
    "${MPI_INCLUDE_PATH}"
    "${MPI_CXX_INCLUDE_DIRS}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_compile_definitions(simulateqcd_hisq_dirac_benchmark PRIVATE
    USE_CUDA
    ARCHITECTURE=${ARCHITECTURE}
    GIT_HASH="external-dirac-benchmark"
    HALODEPTH_2=1
    HALODEPTHSPIN_3=1
    DOUBLEPREC=1
    SINGLEPREC=1
    COMP_R18=1
    COMP_U3R14=1
    NSTACKS_1=1
    LAYOUT_ALL=1
    LAYOUT_ODD=1
    LAYOUT_EVEN=1)
if(USE_GPU_AWARE_MPI)
    target_compile_definitions(
        simulateqcd_hisq_dirac_benchmark PRIVATE USE_GPU_AWARE_MPI)
endif()
if(USE_GPU_P2P)
    target_compile_definitions(
        simulateqcd_hisq_dirac_benchmark PRIVATE USE_GPU_P2P)
endif()
if(COMMS_STREAMS)
    target_compile_definitions(
        simulateqcd_hisq_dirac_benchmark PRIVATE COMMS_STREAMS)
endif()
if(USE_TILED_MULTIRHS)
    target_compile_definitions(
        simulateqcd_hisq_dirac_benchmark PRIVATE USE_TILED_MULTIRHS)
endif()
set_target_properties(simulateqcd_hisq_dirac_benchmark PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/reference")
