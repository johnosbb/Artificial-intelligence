#!/bin/bash

MILKV_BOARD="milkv-duos-glibc-arm64-sd"
MILKV_BOARD_CONFIG="device/${MILKV_BOARD}/boardconfig.sh"
MILKV_IMAGE_CONFIG="device/${MILKV_BOARD}/genimage.cfg"

echo "Disable any Conda Environments"
conda deactivate

TOP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd ${TOP_DIR}

function print_info()
{
    printf "\e[1;32m%s\e[0m\n" "$1"
}

function print_err()
{
    printf "\e[1;31mError: %s\e[0m\n" "$1"
}

function get_toolchain()
{
    if [ ! -d host-tools ]; then
        print_info "DEBUG: Toolchain does not exist, download it now..."
        
        toolchain_url="https://github.com/milkv-duo/host-tools.git"
        echo "toolchain_url: ${toolchain_url}"
        
        git clone ${toolchain_url}
        if [ $? -ne 0 ]; then
            echo "DEBUG: Failed to download ${toolchain_url}!"
            exit 1
        fi
    fi
}

function prepare_env()
{
    source ${MILKV_BOARD_CONFIG}
    
    source build/${MV_BUILD_ENV} > /dev/null 2>&1
    defconfig ${MV_BOARD_LINK} > /dev/null 2>&1
    
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"
    if [ -z "${OUTPUT_DIR// }" ]; then
        print_err "DEBUG: OUTPUT_DIR is not assigned, please check!"
        exit 1
    fi
    
    if [ "${STORAGE_TYPE}" == "sd" ]; then
        if [ ! -f ${MILKV_IMAGE_CONFIG} ]; then
            print_err "DEBUG: ${MILKV_IMAGE_CONFIG} not found!"
            exit 1
        fi
    fi
}

function milkv_build()
{
    echo "DEBUG: Starting build..."
    
    # Skipping cleaning steps here for a faster rebuild
    # Commenting out clean_all and old image cleanup steps
    
    echo "DEBUG: Starting full build..."
    build_all
    
    if [ $? -eq 0 ]; then
        print_info "Build board ${MILKV_BOARD} success!"
    else
        print_err "Build board ${MILKV_BOARD} failed!"
        exit 1
    fi
}

function milkv_pack_sd()
{
    pack_sd_image
    
    [ ! -d out ] && mkdir out
    
    img_in="${OUTPUT_DIR}/${MILKV_BOARD}.img"
    img_out="${MILKV_BOARD}_`date +%Y-%m%d-%H%M`.img"
    
    if [ -f "${img_in}" ]; then
        mv ${img_in} out/${img_out}
        print_info "DEBUG: Create SD image successful: out/${img_out}"
    else
        print_err "DEBUG: Create SD image failed!"
        exit 1
    fi
}

function milkv_pack_emmc()
{
    [ ! -d out ] && mkdir out
    
    img_in="${OUTPUT_DIR}/upgrade.zip"
    img_out="${MILKV_BOARD}_`date +%Y-%m%d-%H%M`.zip"
    
    if [ -f "${img_in}" ]; then
        mv ${img_in} out/${img_out}
        print_info "DEBUG: Create eMMC image successful: out/${img_out}"
    else
        print_err "DEBUG: Create eMMC image failed!"
        exit 1
    fi
}

function milkv_pack_nor_nand()
{
    [ ! -d out ] && mkdir out
    
    if [ -f "${OUTPUT_DIR}/upgrade.zip" ]; then
        img_out_patch=${MILKV_BOARD}-`date +%Y%m%d-%H%M`
        mkdir -p out/$img_out_patch
        
        if [ "${STORAGE_TYPE}" == "spinor" ]; then
            cp ${OUTPUT_DIR}/fip.bin out/$img_out_patch
            cp ${OUTPUT_DIR}/*.spinor out/$img_out_patch
        else
            cp ${OUTPUT_DIR}/fip.bin out/$img_out_patch
            cp ${OUTPUT_DIR}/*.spinand out/$img_out_patch
        fi
        
        echo "Copy all to a blank tf card, power on and automatically download firmware to NOR or NAND in U-boot." >> out/$img_out_patch/how_to_download.txt
        print_info "Create spinor/nand img successful: ${img_out_patch}"
    else
        print_err "Create spinor/nand img failed!"
        exit 1
    fi
}

function milkv_pack()
{
    if [ "${STORAGE_TYPE}" == "sd" ]; then
        milkv_pack_sd
        elif [ "${STORAGE_TYPE}" == "emmc" ]; then
        milkv_pack_emmc
    else
        milkv_pack_nor_nand
    fi
}

function build_info()
{
    print_info "Target Board: ${MILKV_BOARD}"
    print_info "Target Board Storage: ${STORAGE_TYPE}"
    print_info "Target Board Config: ${MILKV_BOARD_CONFIG}"
    if [ "${STORAGE_TYPE}" == "sd" ]; then
        print_info "Target Image Config: ${MILKV_IMAGE_CONFIG}"
    fi
    print_info "Build tdl-sdk: ${TPU_REL}"
}

echo "DEBUG: Preparing Environment for ${MILKV_BOARD}"
prepare_env

echo "DEBUG: Build Info for ${MILKV_BOARD}"
build_info


(cd /mnt/500GB/MilkVDuoS/duo-buildroot-sdk-v2/buildroot-2024.02/output/milkv-duos-glibc-arm64-sd;make rootfs-tar)
echo "DEBUG: Building for ${MILKV_BOARD}"
milkv_build

echo "DEBUG: Packing image for ${MILKV_BOARD}"
milkv_pack
