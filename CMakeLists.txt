cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(face_recognition)

find_package(Torch REQUIRED)

add_executable(face_recognition main.cpp)
target_link_libraries(face_recognition ${TORCH_LIBRARIES})

