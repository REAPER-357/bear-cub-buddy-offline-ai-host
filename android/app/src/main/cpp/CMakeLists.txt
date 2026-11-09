cmake_minimum_required(VERSION 3.22.1)

# Always optimise: llama.cpp is unusably slow in a Debug build.
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

project(llamabridge LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- llama.cpp options (static, CPU only, no extras)
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
# Your phone (MT6833): FP16 + dot-product yes, I8MM/SVE no.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)

add_subdirectory(llama.cpp)

add_library(llamabridge SHARED llama_jni.cpp)
target_include_directories(llamabridge PRIVATE llama.cpp/include llama.cpp/ggml/include)
target_link_libraries(llamabridge llama ggml android log)
