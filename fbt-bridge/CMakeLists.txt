cmake_minimum_required(VERSION 3.16)
project(dinn_fbt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenFHE REQUIRED)

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/schemelet)
include_directories(${OpenFHE_INCLUDE}/third-party/include)

add_executable(dinn_fbt dinn_fbt.cpp)

target_include_directories(dinn_fbt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party
)

target_link_libraries(dinn_fbt ${OpenFHE_LIBRARIES})

target_compile_options(dinn_fbt PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-parameter
)
