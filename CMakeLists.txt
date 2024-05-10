cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER $ENV{HOME}/env/miniconda3/envs/bubblebandit/bin/nvcc)
project(cmake_template VERSION 0.1.0 LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Protobuf_DIR "$ENV{HOME}/env/grpc/lib/cmake/protobuf")
include(common.cmake)

# message("${CMAKE_CURRENT_BINARY_DIR}")

get_filename_component(task_proto "task.proto" ABSOLUTE)
get_filename_component(task_proto_path "${task_proto}" PATH)
get_filename_component(scheduler_proto "scheduler.proto" ABSOLUTE)
get_filename_component(scheduler_proto_path "${scheduler_proto}" PATH)

message("${task_proto}")
message("${task_proto_path}")
message("${scheduler_proto}")
message("${scheduler_proto_path}")

# Generated sources
set(task_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/task.pb.cc")
set(task_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/task.pb.h")
set(task_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/task.grpc.pb.cc")
set(task_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/task.grpc.pb.h")
set(scheduler_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.pb.cc")
set(scheduler_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.pb.h")
set(scheduler_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.grpc.pb.cc")
set(scheduler_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/scheduler.grpc.pb.h")
add_custom_command(
      OUTPUT "${task_proto_srcs}" "${task_proto_hdrs}" "${task_grpc_srcs}" "${task_grpc_hdrs}" "${scheduler_proto_srcs}" "${scheduler_proto_hdrs}" "${scheduler_grpc_srcs}" "${scheduler_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${task_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${task_proto}" "${scheduler_proto}"
      DEPENDS "${task_proto}" "${scheduler_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# task_grpc_proto
add_library(task_grpc_proto
  ${task_grpc_srcs}
  ${task_grpc_hdrs}
  ${task_proto_srcs}
  ${task_proto_hdrs})
target_link_libraries(task_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# scheduler_grpc_proto
add_library(scheduler_grpc_proto
  ${scheduler_grpc_srcs}
  ${scheduler_grpc_hdrs}
  ${scheduler_proto_srcs}
  ${scheduler_proto_hdrs})
target_link_libraries(scheduler_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})


add_executable(main main.cc)
set_target_properties(main PROPERTIES
CUDA_SEPARABLE_COMPILATION ON)
# target_include_directories(cmake_template PRIVATE ${PROTOBUF_INCLUDE_DIR})
target_include_directories(main PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(main PRIVATE task_grpc_proto)

add_executable(bfs_linear_base
src/bfs/linear_base.cu
src/bfs/main.cc
src/bfs/verifier.cc
)
set_target_properties(bfs_linear_base PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(bfs_linear_base
PRIVATE include
)

add_executable(bfs_side_task
src/bfs/bfs_side_task.cu
src/bfs/verifier.cc
)
set_target_properties(bfs_side_task PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(bfs_side_task
PRIVATE include
PRIVATE $ENV{HOME}/env/miniconda3/envs/bubblebandit/include/cub
)
target_link_libraries(bfs_side_task
PRIVATE argparse
PRIVATE task_grpc_proto
PRIVATE scheduler_grpc_proto
)

add_executable(pr_side_task
src/pr/pr_side_task.cu
src/pr/verifier.cc
)
set_target_properties(pr_side_task PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(pr_side_task
PRIVATE include
PRIVATE $ENV{HOME}/env/miniconda3/envs/bubblebandit/include/cub
)
target_link_libraries(pr_side_task
PRIVATE argparse
PRIVATE task_grpc_proto
PRIVATE scheduler_grpc_proto
)

add_executable(sgd_side_task
src/sgd/sgd_side_task.cu
)
set_target_properties(sgd_side_task PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(sgd_side_task
PRIVATE include
PRIVATE $ENV{HOME}/env/miniconda3/envs/bubblebandit/include/cub
)
target_link_libraries(sgd_side_task
PRIVATE argparse
PRIVATE task_grpc_proto
PRIVATE scheduler_grpc_proto
)

# Targets task_(client|server)
# foreach(_target
#   task_client task_server)
#   add_executable(${_target}
#     "${_target}.cc")
#   target_link_libraries(${_target}
#     task_grpc_proto
#     task_helper
#     ${_REFLECTION}
#     ${_GRPC_GRPCPP}
#     ${_PROTOBUF_LIBPROTOBUF})
# endforeach()