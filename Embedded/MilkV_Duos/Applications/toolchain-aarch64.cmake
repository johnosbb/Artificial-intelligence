# toolchain-aarch64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Adjust this path to match your actual toolchain location
set(TOOLCHAIN_PATH "/mnt/500GB/MilkVDuoS/duo-buildroot-sdk-v2/buildroot-2024.02/output/milkv-duos-glibc-arm64-sd/host")

set(CMAKE_SYSROOT "${TOOLCHAIN_PATH}/aarch64-buildroot-linux-gnu/sysroot")

set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH "${TOOLCHAIN_PATH}/aarch64-buildroot-linux-gnu/sysroot")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Optional: Specify linker and archiver
set(CMAKE_AR ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-ar)
set(CMAKE_RANLIB ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-ranlib)
set(CMAKE_STRIP ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-strip)

# Ensure we use the correct linker flags
set(CMAKE_EXE_LINKER_FLAGS "--sysroot=${CMAKE_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS "--sysroot=${CMAKE_SYSROOT}")
