cmake_minimum_required(VERSION 3.10)
project(DiNN_OpenFHE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE
find_package(OpenFHE REQUIRED)

# This is the "magic" line that usually handles standard installs
include_directories(${OpenFHE_INCLUDE})

# These are the specific sub-folders OpenFHE headers look for
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/schemelet)
include_directories(${OpenFHE_INCLUDE}/third-party/include)

# Create the executable
add_executable(dinn_inference testar_com_64GB.cpp)

# Link OpenFHE libraries
target_link_libraries(dinn_inference ${OpenFHE_LIBRARIES})
