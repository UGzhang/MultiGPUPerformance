cmake_minimum_required(VERSION 3.14.5)
project(boltzmann-mpi)

set(CMAKE_CXX_STANDARD 17)

# Define the executable
add_executable(boltzmann-mpi d2q9-bgk.c)

# Set the output directory for the executable
# Output to the parent directory of the current build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../../)



# Find MPI
include(FindMPI)
if (MPI_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(boltzmann-mpi ${MPI_C_LIBRARIES} m)  # Link against MPI and math library
else (MPI_FOUND)
    message(FATAL_ERROR "Unable to find the MPI libraries required to build ${CMAKE_PROJECT_NAME}")
endif (MPI_FOUND)