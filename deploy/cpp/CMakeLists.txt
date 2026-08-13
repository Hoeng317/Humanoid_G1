cmake_minimum_required(VERSION 3.20)
project(humanoid_g1_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

find_package(Torch REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(humanoid_g1_controller_lib src/controller.cpp)
target_include_directories(humanoid_g1_controller_lib PUBLIC include)

add_executable(g1_controller src/main.cpp)
target_link_libraries(g1_controller PRIVATE
  humanoid_g1_controller_lib
  "${TORCH_LIBRARIES}"
  unitree_sdk2
  yaml-cpp
  pthread
)
target_compile_options(g1_controller PRIVATE ${TORCH_CXX_FLAGS})
set_target_properties(g1_controller PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../generated"
  BUILD_RPATH "${CMAKE_CURRENT_LIST_DIR}/../../.local/unitree_sdk2/lib"
)

