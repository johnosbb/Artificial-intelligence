cmake_minimum_required(VERSION 3.10)
project(HelloWorld C)

# Set the C standard to use
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Specify the source file
add_executable(hello_world helloworld.c)

# Install target (optional)
install(TARGETS hello_world RUNTIME DESTINATION bin)
