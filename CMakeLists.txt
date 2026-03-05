cmake_minimum_required(VERSION 3.10)
project(llamastream)

set(CMAKE_CXX_STANDARD 20)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(AWSSDK QUIET COMPONENTS bedrock-runtime)

include_directories(.)

# Base sources (always built)
set(LLAMASTREAM_SOURCES
    ollamastream.cpp
    ollamastream.h
    llmstream.h)

# Add Bedrock support if AWS SDK is available
if(AWSSDK_FOUND)
    message(STATUS "AWS SDK found, building with Bedrock support")
    list(APPEND LLAMASTREAM_SOURCES
        bedrockstream.cpp
        bedrockstream.h)
else()
    message(STATUS "AWS SDK not found, building without Bedrock support")
endif()

# Add llama.cpp support if available
find_path(LLAMA_INCLUDE_DIR llama.h)

if(LLAMA_INCLUDE_DIR)
    message(STATUS "llama.cpp found, building with llamastream support")
    list(APPEND LLAMASTREAM_SOURCES
        llamastream.cpp
        llamastream.h)
endif()

add_library(llamastream STATIC ${LLAMASTREAM_SOURCES})

target_link_libraries(llamastream PRIVATE nlohmann_json::nlohmann_json CURL::libcurl)

if(AWSSDK_FOUND)
    target_link_libraries(llamastream PRIVATE ${AWSSDK_LINK_LIBRARIES})
endif()
