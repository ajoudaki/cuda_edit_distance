cmake_minimum_required(VERSION 3.17)
project(untitled CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(main main.cu)

add_executable(par_edit par_ed.cu)

add_executable(query query.cu )

add_executable(trasn transfer.cu)

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)