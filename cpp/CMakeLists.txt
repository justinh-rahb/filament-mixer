cmake_minimum_required(VERSION 3.14)
project(filament_mixer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(filament_mixer_example example.cpp)
target_include_directories(filament_mixer_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
