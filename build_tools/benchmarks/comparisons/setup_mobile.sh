#!/bin/bash

# Run commands below on the workstation that the phone is attached to.
# Prerequisites:
#   Manual installations of the Android NDK and ADB are needed. See https://google.github.io/iree/building-from-source/android/#install-android-ndk-and-adb for instructions.
#   Manual installations of the Termux App and python are needed on the Android device. See README.md for instructions.

GPU_TYPE="mali"
#GPU_TYPE="andreno"

# Create root dir.
ROOT_DIR=/tmp/mobilebert_benchmarks
mkdir ${ROOT_DIR}
mkdir ${ROOT_DIR}/models
mkdir ${ROOT_DIR}/models/tflite
mkdir ${ROOT_DIR}/models/iree
mkdir ${ROOT_DIR}/setup
mkdir ${ROOT_DIR}/test_data
mkdir ${ROOT_DIR}/output

wget https://storage.googleapis.com/iree-model-artifacts/tflite_squad_test_data.zip -O /tmp/tflite_squad_test_data.zip
unzip /tmp/tflite_squad_test_data.zip -d ${ROOT_DIR}/test_data/
wget https://storage.googleapis.com/iree-model-artifacts/mobilebert_float_384_gpu.tflite -O ${ROOT_DIR}/models/tflite/mobilebert_float_384_gpu.tflite

# Build IREE source.
SOURCE_DIR=/tmp/github
mkdir ${SOURCE_DIR}
cd ${SOURCE_DIR}

#git clone https://github.com/google/iree.git
git clone https://github.com/mariecwhite/iree.git

cd iree
git checkout origin/comparisons

cp ${SOURCE_DIR}/iree/build_tools/benchmarks/set_adreno_gpu_scaling_policy.sh ${ROOT_DIR}/setup/
cp ${SOURCE_DIR}/iree/build_tools/benchmarks/set_android_scaling_governor.sh ${ROOT_DIR}/setup/
cp ${SOURCE_DIR}/iree/build_tools/benchmarks/set_pixel6_gpu_scaling_policy.sh ${ROOT_DIR}/setup/

git submodule update --init
cmake -GNinja -B ../iree-build/ -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIREE_ENABLE_LLD=ON
cmake --build ../iree-build/

export CC=clang
export CXX=clang++
python configure_bazel.py

cd integrations/tensorflow
bazel build -c opt iree_tf_compiler:iree-import-tflite

IREE_COMPILE_PATH=${SOURCE_DIR}/iree-build/iree/tools/iree-compile

TFLITE_MODEL_DIR=${ROOT_DIR}/models/tflite
IREE_MODEL_DIR=${ROOT_DIR}/models/iree
mkdir -p ${IREE_MODEL_DIR}/vulkan
mkdir -p ${IREE_MODEL_DIR}/dylib

MODEL_NAME="mobilebert_float_384_gpu"
bazel-bin/iree_tf_compiler/iree-import-tflite ${TFLITE_MODEL_DIR}/${MODEL_NAME}.tflite -o ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir
echo "Compiling ${MODEL_NAME}.vmfb for aarch64..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=dylib-llvm-aot \
  --iree-llvm-target-triple=aarch64-none-linux-android29 \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/dylib/${MODEL_NAME}.vmfb

echo "Compiling ${MODEL_NAME}_mmt4d.vmfb for aarch64..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=dylib-llvm-aot \
  --iree-llvm-target-triple=aarch64-none-linux-android29 \
  "--iree-flow-mmt4d-target-options=arch=aarch64 features=+dotprod" \
  --iree-llvm-target-cpu-features=+dotprod \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/dylib/${MODEL_NAME}_mmt4d.vmfb

if [[ ${GPU_TYPE} = "mali" ]]
then
echo "Compiling ${MODEL_NAME}.vmfb for vulkan mali..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vulkan-spirv \
  --iree-vulkan-target-triple=valhall-unknown-android11 \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/vulkan/${MODEL_NAME}.vmfb
else
echo "Compiling ${MODEL_NAME}.vmfb for vulkan adreno..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vulkan-spirv \
  --iree-vulkan-target-triple=adreno-unknown-android11 \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/vulkan/${MODEL_NAME}.vmfb
fi

# Cross-compile IREE benchmark binary.
cd ${SOURCE_DIR}/iree
cmake -GNinja -B ../iree-build/ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .
cmake --build ../iree-build/ --target install

rm -rf ${SOURCE_DIR}/iree-build-android

cmake -GNinja -B ../iree-build-android/ \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT="$PWD/../iree-build/install" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM="android-29" \
  -DIREE_BUILD_COMPILER=OFF \
  .
cmake --build ../iree-build-android/
cp ${SOURCE_DIR}/iree-build-android/iree/tools/iree-benchmark-module ${ROOT_DIR}/

# Cross-compile TFLite benchmark binary.
sudo apt-get install libgles2-mesa-dev

export CC=clang
export CXX=clang++

cd ${SOURCE_DIR}
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
# Select defaults. Answer Yes to configuring ./WORKSPACE for Android builds.
# Use Version 21 for Android NDK, 29 for Android SDK.
python configure.py
bazel build -c opt --config=android_arm64 \
  --copt="-Wno-error=implicit-function-declaration" \
  tensorflow/lite/tools/benchmark:benchmark_model

cp ${SOURCE_DIR}/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model ${ROOT_DIR}/

# Push benchmarking artifacts to device.
adb shell rm -r /data/local/tmp/mobilebert_benchmarks
adb push ${ROOT_DIR} /data/local/tmp

DEVICE_ROOT_DIR=/data/local/tmp/mobilebert_benchmarks
adb shell chmod +x ${DEVICE_ROOT_DIR}/benchmark_model
adb shell chmod +x ${DEVICE_ROOT_DIR}/iree-benchmark-module

# Setup device.
adb shell "su root sh ${DEVICE_ROOT_DIR}/setup/set_android_scaling_governor.sh performance"

if [[ ${GPU_TYPE} = "mali" ]]
then
adb shell "su root sh ${DEVICE_ROOT_DIR}/setup/set_pixel6_gpu_scaling_policy.sh performance"
else
adb shell "su root sh ${DEVICE_ROOT_DIR}/setup/set_adreno_gpu_scaling_policy.sh performance"
fi

# Run benchmark.
cp ${SOURCE_DIR}/build_tools/benchmarks/comparisons /data/local/tmp
adb shell "su root /data/data/com.termux/files/usr/bin/python /data/local/tmp/comparisons/run_benchmarks.py --device_name=Pixel6  --mode=mobile --base_dir=${DEVICE_ROOT_DIR} --output_dir=${DEVICE_ROOT_DIR}/output"
adb shell cat ${DEVICE_ROOT_DIR}/output/result.csv