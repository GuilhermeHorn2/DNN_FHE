cmake_minimum_required(VERSION 3.10)
project(DiNN_OpenFHE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Borrows only ../src/bench; the poly pipeline is otherwise self-contained.
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

find_package(OpenFHE REQUIRED)

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/schemelet)
include_directories(${OpenFHE_INCLUDE}/third-party/include)

# Bench / build options
# BENCH_ALL=ON forces every BENCH_* flag on and overrides the individual ones.
option(BENCH_ALL       "Enable every BENCH_* flag at once (timers + memory)" OFF)
option(BENCH_TOTAL     "Time the entire program (main start to end)"         OFF)
option(BENCH_INFERENCE "Time the inference path (per-image)"                 OFF)
option(BENCH_BOOTSTRAP "Time activation/refresh blocks (FBT-equivalent)"     OFF)
option(BENCH_LAYERS    "Time each layer phase"                               OFF)
option(BENCH_MEMORY    "Report peak RSS at instrumentation points"           OFF)

if(BENCH_ALL)
    set(BENCH_TOTAL     ON)
    set(BENCH_INFERENCE ON)
    set(BENCH_BOOTSTRAP ON)
    set(BENCH_LAYERS    ON)
    set(BENCH_MEMORY    ON)
endif()

set(BENCH_DEFS "")
foreach(b TOTAL INFERENCE BOOTSTRAP LAYERS MEMORY)
    if(BENCH_${b})
        list(APPEND BENCH_DEFS BENCH_${b})
    endif()
endforeach()

# Static bench library (sources at ../src/bench)
file(GLOB POLYBENCH_SOURCES "${REPO_ROOT}/src/bench/*.cpp")
add_library(polybench STATIC ${POLYBENCH_SOURCES})
target_include_directories(polybench PUBLIC "${REPO_ROOT}/include")
target_compile_definitions(polybench PUBLIC ${BENCH_DEFS})

# Executables
add_executable(main relu_running.cpp)
target_link_libraries(main PRIVATE ${OpenFHE_LIBRARIES} polybench)
target_compile_definitions(main PRIVATE ${BENCH_DEFS})
