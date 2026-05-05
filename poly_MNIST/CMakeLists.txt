cmake_minimum_required(VERSION 3.10)
project(DiNN100_OpenFHE_Poly)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Shared headers (e.g. stb_image.h) live in the repo root's include/ ───────
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories("${REPO_ROOT}/include")

# ── OpenFHE ──────────────────────────────────────────────────────────────────
find_package(OpenFHE REQUIRED)

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/schemelet)
include_directories(${OpenFHE_INCLUDE}/third-party/include)

# Create the executable
add_executable(main main.cpp)

# Link OpenFHE libraries
target_link_libraries(main ${OpenFHE_LIBRARIES})
