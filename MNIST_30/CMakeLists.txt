cmake_minimum_required(VERSION 3.10)
project(DiNN_OpenFHE_Modular)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Shared framework lives one level up (../include, ../src, ../accuracy.cpp).
#     This sub-project only owns its driver (main.cpp), its weights
#     (dinn30_*.csv), and its sample images (img_*.jpg).
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ── OpenFHE ──────────────────────────────────────────────────────────────────
find_package(OpenFHE REQUIRED)

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/schemelet)
include_directories(${OpenFHE_INCLUDE}/third-party/include)

# ── Bench / build options ────────────────────────────────────────────────────
option(BENCH_TOTAL     "Time the entire program (main start to end)"       OFF)
option(BENCH_INFERENCE "Time the inference path inside Network::Run"       OFF)
option(BENCH_BOOTSTRAP "Time activation/refresh blocks (FBT-equivalent)"   OFF)
option(BENCH_LAYERS    "Time each layer's Apply()"                         OFF)
option(BENCH_MEMORY    "Report peak RSS at instrumentation points"         OFF)
option(BUILD_ACCURACY  "Build the accuracy benchmark binary"               OFF)

set(BENCH_DEFS "")
foreach(b TOTAL INFERENCE BOOTSTRAP LAYERS MEMORY)
    if(BENCH_${b})
        list(APPEND BENCH_DEFS BENCH_${b})
    endif()
endforeach()

# ── Static framework library (sources at ../src) ─────────────────────────────
file(GLOB FHEDNN_SOURCES
    "${REPO_ROOT}/src/network/*.cpp"
    "${REPO_ROOT}/src/io/*.cpp"
    "${REPO_ROOT}/src/bench/*.cpp")

add_library(fhednn STATIC ${FHEDNN_SOURCES})
target_include_directories(fhednn PUBLIC "${REPO_ROOT}/include")
target_link_libraries(fhednn PUBLIC ${OpenFHE_LIBRARIES})
target_compile_definitions(fhednn PUBLIC ${BENCH_DEFS})

# ── Executables ──────────────────────────────────────────────────────────────
add_executable(main main.cpp)
target_link_libraries(main PRIVATE fhednn)
target_compile_definitions(main PRIVATE ${BENCH_DEFS})

if(BUILD_ACCURACY)
    add_executable(accuracy "${REPO_ROOT}/accuracy.cpp")
    target_link_libraries(accuracy PRIVATE fhednn)
    target_compile_definitions(accuracy PRIVATE ${BENCH_DEFS})
endif()
