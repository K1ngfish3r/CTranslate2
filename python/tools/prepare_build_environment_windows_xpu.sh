#! /bin/bash

set -e
set -x

# See https://github.com/oneapi-src/oneapi-ci for installer URLs
# Install Intel oneAPI toolkit with DPC++ compiler, MKL, and oneDNN GPU
if [ ! -d "C:/Program Files (x86)/Intel/oneAPI" ]; then
    curl --netrc-optional -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/IRC_NAS/4144bec3-82ce-4672-bd71-5c93a79cd5e7/intel-oneapi-toolkit-2026.1.0.191_offline.exe
    ./webimage.exe -s -x -f webimage_extracted --log extract.log
    rm webimage.exe
    # List available components for debugging (can remove once stable)
    echo "=== Available components ==="
    ./webimage_extracted/bootstrapper.exe -s --list-components 2>/dev/null || true
    echo "=== End components ==="
    # Install DPC++ compiler, MKL dev, and oneDNN (TBB pulled as dep of DNNL)
    # Note: oneCCL is not available in this toolkit — XPU tensor parallel is
    # not supported on Windows builds.
    ./webimage_extracted/bootstrapper.exe -s --action install \
        --components="intel.oneapi.win.cpp-dpcpp-common:intel.oneapi.win.mkl.devel:intel.oneapi.win.dnnl:intel.oneapi.win.mpi.devel" \
        --eula=accept \
        -p=NEED_VS2017_INTEGRATION=0 \
        -p=NEED_VS2019_INTEGRATION=0 \
        --log-dir=.
fi

# Install Level Zero GPU loader (optional — needed for GPU runtime, not build)
# The SYCL runtime uses Level Zero to talk to Intel GPUs at runtime.
curl --netrc-optional -L -nv -o level-zero.zip https://github.com/oneapi-src/level-zero/releases/download/v1.28.2/level-zero-win-sdk-1.28.2.zip
unzip -o level-zero.zip -d "C:/Program Files (x86)/Intel/oneAPI/level-zero" 2>/dev/null \
    || echo "WARNING: Level Zero SDK not installed. GPU runtime may be unavailable."
rm -f level-zero.zip

NPROC=$(nproc)

# Set INTEL_ROOT so CMake's find_path/find_library can locate oneAPI components
export INTEL_ROOT="C:/Program Files (x86)/Intel"
ONEAPI_ROOT="${INTEL_ROOT}/oneAPI"

# Resolve Intel oneAPI version directory (the "latest" symlink may not exist).
# The installer creates a versioned directory like "2026.1".
if [ -f "${ONEAPI_ROOT}/compiler/latest/bin/icpx.exe" ]; then
    ONEAPI_COMPILER_VER="latest"
elif [ -f "${ONEAPI_ROOT}/compiler/2026.1/bin/icpx.exe" ]; then
    ONEAPI_COMPILER_VER="2026.1"
else
    # Auto-detect: find the first version directory containing icpx.exe
    ONEAPI_COMPILER_DIR=$(ls -d "${ONEAPI_ROOT}/compiler/20"* 2>/dev/null | head -1)
    ONEAPI_COMPILER_VER=$(basename "$ONEAPI_COMPILER_DIR" 2>/dev/null)
    if [ -z "$ONEAPI_COMPILER_VER" ] || [ ! -f "${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/bin/icpx.exe" ]; then
        echo "FATAL: Intel DPC++ compiler (icpx.exe) not found under ${ONEAPI_ROOT}/compiler/"
        echo "=== Compiler directory contents ==="
        ls -R "${ONEAPI_ROOT}/compiler/" 2>/dev/null || true
        echo "=== Try finding icpx.exe ==="
        find "${ONEAPI_ROOT}/compiler/" -name 'icpx*' -o -name 'dpcpp*' 2>/dev/null || true
        exit 1
    fi
fi

# Resolve DNNL version directory similarly
if [ -d "${ONEAPI_ROOT}/dnnl/latest" ]; then
    ONEAPI_DNNL_VER="latest"
elif [ -d "${ONEAPI_ROOT}/dnnl/2026.1" ]; then
    ONEAPI_DNNL_VER="2026.1"
else
    ONEAPI_DNNL_DIR_FOUND=$(ls -d "${ONEAPI_ROOT}/dnnl/20"* 2>/dev/null | head -1)
    ONEAPI_DNNL_VER=$(basename "$ONEAPI_DNNL_DIR_FOUND" 2>/dev/null)
fi

# Set compiler to Intel DPC++ (icpx for SYCL compilation mode)
export C_COMPILER="${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/bin/icx.exe"
export CXX_COMPILER="${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/bin/icpx.exe"

# On Windows, CMake defaults to the Visual Studio generator which ignores
# CMAKE_CXX_COMPILER and falls back to MSVC. Use Ninja so that the Intel
# DPC++ compiler (icpx) is actually used for SYCL compilation.
export CMAKE_GENERATOR="Ninja"

# Set up Intel compiler environment for bash on Windows.
# The Intel compiler driver (icx/icpx) needs LIB and INCLUDE to find its own
# runtime libraries (libircmt.lib, etc.). On Linux, setvars.sh handles this;
# on Windows bash, we capture the env from vars.bat via cmd.
_COMPILER_ENV="${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/env/vars.bat"
if [ -f "$_COMPILER_ENV" ]; then
    cmd.exe /c "call \"$_COMPILER_ENV\" > nul && set" 2>/dev/null | tr -d '\r' > /tmp/intel_env
    while IFS='=' read -r key value; do
        case "$key" in
            LIB|INCLUDE|ONEAPI_ROOT|TBB_ROOT|DNNL_ROOT)
                export "$key=$value"
                ;;
        esac
    done < /tmp/intel_env
    rm -f /tmp/intel_env
fi
# Ensure at minimum the compiler lib path is available (fallback)
export LIB="${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/lib${LIB:+;$LIB}"
export INCLUDE="${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/include${INCLUDE:+;$INCLUDE}"

export ONEAPI_COMPILER_LIB="${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/lib"
export ONEAPI_DNNL_DIR="${ONEAPI_ROOT}/dnnl/${ONEAPI_DNNL_VER}"

# Build CTranslate2 with XPU support
mkdir build && cd build

cmake -G "${CMAKE_GENERATOR}" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$CTRANSLATE2_ROOT \
      -DCMAKE_C_COMPILER="${C_COMPILER}" \
      -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_PREFIX_PATH="${ONEAPI_COMPILER_LIB};${ONEAPI_DNNL_DIR}" \
      -DINTEL_ROOT="${INTEL_ROOT}" \
      -DBUILD_CLI=OFF \
      -DWITH_MKL=ON \
      -DWITH_XPU=ON \
      -DOPENMP_RUNTIME=INTEL \
      ..

cmake --build . --config Release --target install --parallel $NPROC --verbose
cd ..
rm -rf build

cp README.md python/
cp $CTRANSLATE2_ROOT/bin/ctranslate2.dll python/ctranslate2/

# Bundle MKL OpenMP runtime (same as CUDA Windows build)
cp "${ONEAPI_ROOT}/compiler/${ONEAPI_COMPILER_VER}/bin/libiomp5md.dll" python/ctranslate2/

# SYCL and DNNL GPU DLLs are excluded from the wheel — they must be installed
# via the Intel oneAPI runtime on the target machine (same approach as Linux).
# If the user has oneAPI installed, these will be found from the system path.
echo "Windows XPU wheel built. SYCL/DNNL runtime DLLs are NOT bundled."
echo "Users must install Intel oneAPI runtime on the target machine."
