cmake_minimum_required(VERSION 3.14)
project(scarf_ramulator_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(RAMULATOR_ROOT "" CACHE PATH "Pinned Ramulator 2.1 checkout")
if(NOT EXISTS "${RAMULATOR_ROOT}/libramulator.so")
  message(FATAL_ERROR "Build pinned Ramulator first and set RAMULATOR_ROOT")
endif()

add_executable(ramulator_driver ramulator_driver.cpp)
target_include_directories(ramulator_driver PRIVATE "${RAMULATOR_ROOT}/src")
target_link_libraries(ramulator_driver PRIVATE "${RAMULATOR_ROOT}/libramulator.so")
set_target_properties(ramulator_driver PROPERTIES
  BUILD_RPATH "${RAMULATOR_ROOT}"
  INSTALL_RPATH "${RAMULATOR_ROOT}")
