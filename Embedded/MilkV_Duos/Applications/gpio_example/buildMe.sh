BUILD_DIRECTORY=build_release
mkdir -p $BUILD_DIRECTORY
SOURCE_DIRECTORY=.
rm -rf $BUILD_DIRECTORY/*
CMAKE=/usr/bin/cmake
$CMAKE  -DCMAKE_BUILD_TYPE=Release  -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake -B$BUILD_DIRECTORY -H$SOURCE_DIRECTORY
$CMAKE --build $BUILD_DIRECTORY