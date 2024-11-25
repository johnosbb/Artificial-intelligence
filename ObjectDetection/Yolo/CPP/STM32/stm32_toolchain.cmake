# toolchain.cmake

# Define the toolchain name as a variable (replace this with your toolchain name)
SET(TOOLCHAIN_NAME "arm-linux")

# Set the toolchain prefix
SET(TOOLCHAIN_PREFIX "/mnt/500GB/STM32MP157D/") 

# Set the system name and architecture
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)

# Specify the cross-compiler toolchain binaries using the TOOLCHAIN_PREFIX and TOOLCHAIN_NAME
SET(CMAKE_C_COMPILER "${TOOLCHAIN_PREFIX}/buildroot/output/host/bin/${TOOLCHAIN_NAME}-gcc")
SET(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}/buildroot/output/host/bin/${TOOLCHAIN_NAME}-g++")
SET(CMAKE_AR "${TOOLCHAIN_PREFIX}/buildroot/output/host/bin/${TOOLCHAIN_NAME}-ar")
SET(CMAKE_ASM_COMPILER "${TOOLCHAIN_PREFIX}/buildroot/output/host/bin/${TOOLCHAIN_NAME}-gcc")

# Specify the system root directory (this is the root filesystem of the target system)
SET(CMAKE_FIND_ROOT_PATH "${TOOLCHAIN_PREFIX}/buildroot/output/host/arm-buildroot-linux-gnueabihf/sysroot")

# Print the sysroot location
message(STATUS "Sysroot location for this Toolchain: ${CMAKE_FIND_ROOT_PATH}")

# Direct CMake to search for headers and libraries in the sysroot first
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Specify the build type (optional) : for Debug change this to SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)

# Specify linker flags if necessary (optional)
SET(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

