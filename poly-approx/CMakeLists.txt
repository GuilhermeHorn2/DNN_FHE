cmake_minimum_required(VERSION 3.16)
project(dinn_chebyshev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenFHE REQUIRED)

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/schemelet)
include_directories(${OpenFHE_INCLUDE}/third-party/include)

add_executable(dinn_chebyshev dinn_chebyshev.cpp)

target_include_directories(dinn_chebyshev PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party
)

target_link_libraries(dinn_chebyshev ${OpenFHE_LIBRARIES})

target_compile_options(dinn_chebyshev PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-parameter
)
