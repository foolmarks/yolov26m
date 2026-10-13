cmake_minimum_required(VERSION 3.16)
project(usb_insight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

add_executable(usb_insight src/main.cpp)
target_link_libraries(usb_insight PRIVATE SimaNeat::sima_neat PkgConfig::OPENCV)
