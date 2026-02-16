cmake_minimum_required(VERSION 3.10)
project(llamastream)

set(CMAKE_CXX_STANDARD 20)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(.)

# Check if llama.cpp is available
find_path(LLAMA_INCLUDE_DIR llama.h)

if(LLAMA_INCLUDE_DIR)
    message(STATUS "llama.cpp found, building full llamastream library")
    add_library(llamastream STATIC
            ollamastream.cpp
            ollamastream.h
            llamastream.cpp
            llamastream.h)
else()
    message(STATUS "llama.cpp not found, building ollamastream only")
    add_library(llamastream STATIC
            ollamastream.cpp
            ollamastream.h)
endif()

target_link_libraries(llamastream PRIVATE nlohmann_json::nlohmann_json CURL::libcurl)
