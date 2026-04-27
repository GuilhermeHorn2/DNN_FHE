cmake_minimum_required(VERSION 3.16)
project(HENeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find OpenFHE ──────────────────────────────────────────────────────────────
# Set OPENFHE_ROOT if OpenFHE is installed in a non-standard location, e.g.:
#   cmake -DOPENFHE_ROOT=/usr/local ..
find_package(OpenFHE REQUIRED
    HINTS ${OPENFHE_ROOT}
          /usr/local
          /opt/openfhe
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(he_neural_network he_neural_network.cpp)

target_include_directories(he_neural_network PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
)

target_link_libraries(he_neural_network PRIVATE
    OPENFHEcore
    OPENFHEpke
    OPENFHEbinfhe
)

# ── Compiler flags ────────────────────────────────────────────────────────────
target_compile_options(he_neural_network PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wno-unused-parameter
)

message(STATUS "OpenFHE version : ${OpenFHE_VERSION}")
message(STATUS "OpenFHE include : ${OpenFHE_INCLUDE}")
message(STATUS "OpenFHE libs    : ${OpenFHE_LIBRARIES}")
