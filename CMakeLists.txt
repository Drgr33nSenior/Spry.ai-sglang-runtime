cmake_minimum_required(VERSION 3.24)
project(SpryUltraQuant VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(SPRY_BUILD_HIP "Build the experimental gfx1201 HIP library (no execution)" OFF)

set(spry_identity_inputs
  include/spry/codec.h include/spry/runtime.h native/codec.cpp
  native/runtime_cpu.cpp native/runtime_hip.hip native/build_identity.cpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${spry_identity_inputs})
set(spry_source_identity "")
foreach(input IN LISTS spry_identity_inputs)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${input}" input_sha)
  string(APPEND spry_source_identity "${input}:${input_sha}\n")
endforeach()
string(SHA256 spry_source_sha "${spry_source_identity}")

add_library(spry_uq_cpu SHARED native/codec.cpp native/runtime_cpu.cpp native/build_identity.cpp)
target_include_directories(spry_uq_cpu PUBLIC include)
target_compile_options(spry_uq_cpu PRIVATE -Wall -Wextra -Werror -pedantic -ffp-contract=off)
string(SHA256 spry_cpu_build_id "${spry_source_sha}|${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_BUILD_TYPE}|${CMAKE_CXX_FLAGS}|cpu|fp-contract=off")
target_compile_definitions(spry_uq_cpu PRIVATE SPRY_UQ_BUILD_ID="${spry_cpu_build_id}" SPRY_UQ_BUILD_TARGET="cpu")

if(SPRY_BUILD_HIP)
  if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1201")
    message(FATAL_ERROR "This experimental target supports exactly gfx1201")
  endif()
  set(CMAKE_HIP_ARCHITECTURES gfx1201 CACHE STRING "Required GPU target")
  include(CheckLanguage)
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR "SPRY_BUILD_HIP=ON requires a prepared ROCm HIP compiler; no GPU build was produced")
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  add_library(spry_uq_hip SHARED native/runtime_hip.hip native/runtime_cpu.cpp native/build_identity.cpp)
  set_target_properties(spry_uq_hip PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  target_include_directories(spry_uq_hip PRIVATE include)
  target_link_libraries(spry_uq_hip PRIVATE hip::host)
  target_compile_options(spry_uq_hip PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
  string(SHA256 spry_hip_build_id "${spry_source_sha}|${CMAKE_HIP_COMPILER_ID}|${CMAKE_HIP_COMPILER_VERSION}|${CMAKE_BUILD_TYPE}|${CMAKE_HIP_FLAGS}|gfx1201|fp-contract=off")
  target_compile_definitions(spry_uq_hip PRIVATE SPRY_UQ_BUILD_ID="${spry_hip_build_id}" SPRY_UQ_BUILD_TARGET="gfx1201")
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  add_test(NAME cpu_reference
    COMMAND "${CMAKE_COMMAND}" -E env
      "SPRY_UQ_CPU_LIBRARY=$<TARGET_FILE:spry_uq_cpu>"
      "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
      "PYTHONDONTWRITEBYTECODE=1"
      "${Python3_EXECUTABLE}" -m unittest discover -s tests -p "test_*.py" -v)
  set_tests_properties(cpu_reference PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
