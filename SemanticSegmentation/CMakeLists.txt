cmake_minimum_required(VERSION 2.8)
project(SemanticSegmentation)

find_package(OpenCV REQUIRED)

if (IE_NOT_FOUND_MESSAGE)
    find_package(InferenceEngine 2.0 QUIET)
    if (NOT(InferenceEngine_FOUND))
        message(FATAL_ERROR ${IE_NOT_FOUND_MESSAGE})
    endif()
else()
    find_package(InferenceEngine 2.0 REQUIRED)
    message(STATUS "InferenceEngine ${InferenceEngine_VERSION} is found in ${InferenceEngine_DIR}")
endif()

if(TARGET IE::ie_cpu_extension)
    add_definitions(-DWITH_EXTENSIONS)
endif()

add_executable(segmentation main.cpp Segmentation.cpp Segmentation.h)

target_link_libraries(
        segmentation
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES}
        #  IE::ie_cpu_extension
)
