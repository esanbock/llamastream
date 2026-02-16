cmake_minimum_required(VERSION 3.10)
project(llamastream)

set(CMAKE_CXX_STANDARD 20)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(.)

add_library(llamastream STATIC
        ollamastream.cpp
        ollamastream.h
        llamastream.cpp
        llamastream.h)

target_link_libraries(llamastream PRIVATE nlohmann_json::nlohmann_json CURL::libcurl)
