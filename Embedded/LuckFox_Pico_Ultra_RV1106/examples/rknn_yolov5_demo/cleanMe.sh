
#!/bin/bash
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
# build
echo "Cleaning file in $ROOT_PWD"
BUILD_DIR=${ROOT_PWD}/build/build_linux_arm

if [[ -d "${BUILD_DIR}" ]]; then
    echo "Removing build files from ${BUILD_DIR}"
    rm -rf ${BUILD_DIR}/*
fi
