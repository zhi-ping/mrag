cmake_minimum_required(VERSION 3.21)

project(mrag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# 作为子项目引入时，关闭不需要的附带目标，避免构建过重。
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Build llama.cpp common helpers" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build llama.cpp tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Build llama.cpp embedded web UI" FORCE)

# 开启 CUDA 加速
set(LLAMA_CUDA ON CACHE BOOL "Force disable LLAMA_CUDA" FORCE)
set(GGML_CUDA ON CACHE BOOL "Force disable GGML_CUDA" FORCE)

FetchContent_Declare(
    llamacpp
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b9245.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(llamacpp nlohmann_json)

add_library(project_dependencies INTERFACE)
target_link_libraries(
    project_dependencies
    INTERFACE
        llama
        nlohmann_json::nlohmann_json
)

# 1. 自动搜索 src 目录下所有的 .cc 和 .h 文件
file(GLOB_RECURSE ALL_SRCS 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)

# 2. 定义可执行文件，使用搜索到的文件列表
add_executable(${PROJECT_NAME} ${ALL_SRCS})

# 3. 包含头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 4. 链接依赖
target_link_libraries(${PROJECT_NAME} PRIVATE project_dependencies)
