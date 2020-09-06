cmake_minimum_required(VERSION 2.8)
project(classification)

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

#file (GLOB IE_HEADERS
#        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
#        )

add_executable(cls_main cls_main.cpp classifications.cpp classifications.h)

target_link_libraries(
        cls_main
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

if (TARGET IE::ie_cpu_extension)
    target_link_libraries(cls_main IE::ie_cpu_extension)
endif ()
