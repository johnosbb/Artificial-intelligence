#!/bin/bash
set -e

# --- RKNN Toolkit Installation Check (UNCHANGED, these were good) ---
# Define RKNN SDK Base Directory (must match CMakeLists.txt for consistency)
RKNN_SDK_BASE_DIR="/mnt/500GB/RKNN"

if [ ! -d "$RKNN_SDK_BASE_DIR" ]; then
    echo "Error: RKNN SDK base directory not found!"
    echo "Expected at: $RKNN_SDK_BASE_DIR"
    echo "Please ensure the RKNN toolkit is installed in this location."
    exit 1
fi

RKNN_API_LIB_DIR="${RKNN_SDK_BASE_DIR}/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api"
if [ ! -d "$RKNN_API_LIB_DIR" ]; then
    echo "Error: RKNN API directory not found!"
    echo "Expected at: $RKNN_API_LIB_DIR"
    echo "This indicates an incomplete or incorrect RKNN toolkit runtime installation."
    exit 1
fi

RKNN_MRT_LIB="${RKNN_API_LIB_DIR}/armhf-uclibc/librknnmrt.so"
if [ ! -f "$RKNN_MRT_LIB" ]; then
    echo "Error: RKNN runtime library (librknnmrt.so) not found!"
    echo "Expected at: $RKNN_MRT_LIB"
    echo "Please ensure the correct RKNN runtime for 'armhf-uclibc' is installed."
    exit 1
fi

THIRDPARTY_INCLUDE_DIR="${RKNN_SDK_BASE_DIR}/rknn-toolkit2/rknpu2/examples/3rdparty"
if [ ! -d "$THIRDPARTY_INCLUDE_DIR" ]; then
    echo "Error: 3rdparty include directory not found!"
    echo "Expected at: $THIRDPARTY_INCLUDE_DIR"
    echo "This directory contains essential headers like stb_image.h."
    exit 1
fi

echo "RKNN toolkit components verified successfully."
echo ""

# --- Toolchain Checks (MODIFIED HERE) ---
TOOLCHAIN_PREFIX="/mnt/buildroot_volume/luckfox-pico"
# RK_RV1106_TOOLCHAIN is the *prefix* to the compiler executables (e.g., -gcc, -g++)
RK_RV1106_TOOLCHAIN=$TOOLCHAIN_PREFIX/sysdrv/source/buildroot/buildroot-2023.02.6/output/host/bin/arm-rockchip830-linux-uclibcgnueabihf

# First, check if the variable itself is empty
if [ -z "$RK_RV1106_TOOLCHAIN" ]; then
    echo "Error: RK_RV1106_TOOLCHAIN environment variable is not set or empty!"
    echo "example:"
    echo "  export RK_RV1106_TOOLCHAIN=<path-to-your-dir/arm-rockchip830-linux-uclibcgnueabihf>"
    exit 1
fi

# Extract the directory containing the toolchain binaries
# This takes the directory part of RK_RV1106_TOOLCHAIN (which should be the 'bin' folder)
TOOLCHAIN_BIN_DIR=$(dirname "$RK_RV1106_TOOLCHAIN")

# Verify that the directory containing the compilers exists
if [ ! -d "$TOOLCHAIN_BIN_DIR" ]; then
    echo "Error: Toolchain binary directory not found!"
    echo "Expected at: $TOOLCHAIN_BIN_DIR"
    echo "Please ensure your cross-toolchain is correctly installed."
    exit 1
fi

# Verify the specific C compiler executable exists
if [ ! -f "${RK_RV1106_TOOLCHAIN}-gcc" ]; then
    echo "Error: C compiler (${RK_RV1106_TOOLCHAIN}-gcc) not found!"
    echo "Please verify your toolchain installation and path. Make sure the 'gcc' executable is present."
    exit 1
fi

# Verify the specific C++ compiler executable exists
if [ ! -f "${RK_RV1106_TOOLCHAIN}-g++" ]; then
    echo "Error: C++ compiler (${RK_RV1106_TOOLCHAIN}-g++) not found!"
    echo "Please verify your toolchain installation and path. Make sure the 'g++' executable is present."
    exit 1
fi

echo "Toolchain components verified successfully."
echo ""

# --- Build Process (UNCHANGED) ---
GCC_COMPILER=$RK_RV1106_TOOLCHAIN

ROOT_PWD=$( cd "$( dirname "$0" )" && cd -P "$( dirname "$SOURCE" )" && pwd )

BUILD_DIR=${ROOT_PWD}/build/build_linux_arm

if [[ ! -d "${BUILD_DIR}" ]]; then
    mkdir -p "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"

echo "Configuring CMake..."
cmake ../.. \
-DCMAKE_C_COMPILER="${GCC_COMPILER}-gcc" \
-DCMAKE_CXX_COMPILER="${GCC_COMPILER}-g++"

echo "Building project..."
make -j4

echo "Installing project..."
make install

cd -

echo "Build process completed successfully!"