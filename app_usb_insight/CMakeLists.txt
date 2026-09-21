cmake_minimum_required(VERSION 3.16)
project(app-usb-insight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(SimaNeat REQUIRED CONFIG)
find_package(Threads REQUIRED)

# The sysroot ships an OpenCVConfig.cmake that references modules the image does
# not install, so resolve OpenCV through pkg-config as the Neat package does.
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

if (TARGET SimaNeat::sima_neat_shared)
  set(NEAT_TARGET SimaNeat::sima_neat_shared)
else()
  set(NEAT_TARGET SimaNeat::sima_neat)
endif()

add_executable(app-usb-insight
  src/main.cpp
  src/scalar_config.cpp
  src/v4l2_capture.cpp
)

target_link_libraries(app-usb-insight
  PRIVATE
    ${NEAT_TARGET}
    PkgConfig::OPENCV
    Threads::Threads
)
