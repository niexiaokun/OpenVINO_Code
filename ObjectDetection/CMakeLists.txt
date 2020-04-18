cmake_minimum_required(VERSION 2.8)
project(ObjectDetection)

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

#file (GLOB IE_HEADERS
#        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
#        )

add_executable(ssd_main ssd_main.cpp base_detector.cpp base_detector.h ssd_detector.cpp ssd_detector.h)
add_executable(yolo_main yolo_main.cpp base_detector.cpp base_detector.h yolo_detector.cpp yolo_detector.h)

target_link_libraries(
        ssd_main
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES}
      #  IE::ie_cpu_extension
)
target_link_libraries(
        yolo_main
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBRARIES}
       # IE::ie_cpu_extension
)