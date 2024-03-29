cmake_minimum_required(VERSION 3.5)

#change name of package to this
project(umkc_mpc_ros)
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies this will allow CPP and python
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TurtleCmd.msg"
 )

# Include Cpp "include" directory
include_directories(include)
# Create Cpp executable
add_executable(cpp_node src/cpp_node.cpp)
ament_target_dependencies(cpp_node rclcpp)
# Install Cpp executables
install(TARGETS
  cpp_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/mpc_turtle.py
  scripts/turtle_mpc_node.py
  scripts/example_offboard.py
  scripts/mpc_quad.py
  scripts/fixedwing_offboard.py
  scripts/logger.py
  scripts/test_script.py
  scripts/turtle_prototype.py
  scripts/fw_airplane_mpc.py
  scripts/effector_node.py
  scripts/effector_frame.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()

#launch
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)