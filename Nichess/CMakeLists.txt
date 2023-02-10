cmake_minimum_required(VERSION 3.14)
project(play_vs_agent)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(nichess-cpp)
add_executable(
  ${PROJECT_NAME}
  1/play_vs_agent.cpp
  1/agent1.cpp
  )
target_link_libraries(play_vs_agent nichess)
