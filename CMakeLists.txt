cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(	main 
				main.cpp 
				FaceDetector/face_detector.cpp 
				FacialLandmarks/facial_landmarks.cpp 
				FacialLandmarks/model.cpp 
				FaceRecognition/face_recognition.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${TORCH_LIBRARIES})

