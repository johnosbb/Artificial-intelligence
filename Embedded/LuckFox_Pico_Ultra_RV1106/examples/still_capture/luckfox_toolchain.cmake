# toolchain.cmake

# Set the toolchain prefix
SET(TOOLCHAIN_PREFIX "/mnt/buildroot_volume/luckfox-pico/")

# Set the system name and architecture
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)
SET(TOOL_CHAIN_LOCATION sysdrv/source/buildroot/buildroot-2023.02.6/output/host/bin)
# Specify the cross-compiler toolchain binaries using the TOOLCHAIN_PREFIX
SET(CMAKE_C_COMPILER "${TOOLCHAIN_PREFIX}/${TOOL_CHAIN_LOCATION}/arm-rockchip830-linux-uclibcgnueabihf-gcc")
SET(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}/${TOOL_CHAIN_LOCATION}/arm-rockchip830-linux-uclibcgnueabihf-g++")
SET(CMAKE_AR "${TOOLCHAIN_PREFIX}/${TOOL_CHAIN_LOCATION}/arm-rockchip830-linux-uclibcgnueabihf-ar")
SET(CMAKE_ASM_COMPILER "${TOOLCHAIN_PREFIX}/${TOOL_CHAIN_LOCATION}/arm-rockchip830-linux-uclibcgnueabihf-as")

# Specify the system root directory (this is the root filesystem of the target system)
SET(CMAKE_FIND_ROOT_PATH "${TOOLCHAIN_PREFIX}/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/arm-buildroot-linux-uclibcgnueabihf/sysroot/")

# Direct CMake to search for headers and libraries in the sysroot first
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Specify the build type (optional)
SET(CMAKE_BUILD_TYPE Release)

# Specify linker flags if necessary (optional)
SET(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
