cmake_minimum_required(VERSION 3.20)
project(humanoid_g1_unitree_mujoco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(UNITREE_MUJOCO "${PROJECT_ROOT}/third_party/unitree_mujoco")
set(MUJOCO_ROOT "${PROJECT_ROOT}/.local/mujoco" CACHE PATH "Local MuJoCo release")
set(OUTPUT_ROOT "${PROJECT_ROOT}/.local/unitree_mujoco")

find_package(unitree_sdk2 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)

file(GLOB MUJOCO_UI_SOURCES
  "${MUJOCO_ROOT}/simulate/glfw_*.cc"
  "${MUJOCO_ROOT}/simulate/platform_*.cc"
  "${MUJOCO_ROOT}/simulate/simulate.cc"
)

add_executable(unitree_mujoco
  "${UNITREE_MUJOCO}/simulate/src/main.cc"
  "${UNITREE_MUJOCO}/simulate/src/joystick/joystick.cc"
  "${UNITREE_MUJOCO}/simulate/src/lodepng/lodepng.cpp"
  ${MUJOCO_UI_SOURCES}
)
target_include_directories(unitree_mujoco PRIVATE
  "${UNITREE_MUJOCO}/simulate/src"
  "${UNITREE_MUJOCO}/simulate/src/joystick"
  "${UNITREE_MUJOCO}/simulate/src/lodepng"
  "${MUJOCO_ROOT}/include"
  "${MUJOCO_ROOT}/simulate"
)
target_link_libraries(unitree_mujoco PRIVATE
  "${MUJOCO_ROOT}/lib/libmujoco.so.3.3.6"
  glfw
  yaml-cpp
  unitree_sdk2
  Boost::program_options
  fmt::fmt
  pthread
)
set_target_properties(unitree_mujoco PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_ROOT}/bin"
  BUILD_RPATH "${MUJOCO_ROOT}/lib;${PROJECT_ROOT}/.local/unitree_sdk2/lib"
)
file(MAKE_DIRECTORY "${OUTPUT_ROOT}")
configure_file("${PROJECT_ROOT}/configs/deploy/mujoco.yaml" "${OUTPUT_ROOT}/config.yaml" COPYONLY)
