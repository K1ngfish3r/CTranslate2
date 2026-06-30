#! /bin/bash

set -ex

pip install "cmake==3.22.*"

# Free disk space
rm -rf /host/usr/local/lib/{android,node_modules}
rm -rf /host/usr/local/.ghcup
rm -rf /host/usr/local/share/{powershell,chromium}
rm -rf /host/usr/local/julia*
rm -rf /host/usr/share/{dotnet,swift}
rm -rf /host/usr/share/az_*
rm -rf /host/usr/lib/{jvm,google-cloud-sdk}
rm -rf /host/opt/hostedtoolcache/{CodeQL,go,node,Ruby}
rm -rf /host/opt/{microsoft,az,google}
df -h

export LIBRARY_PATH="/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14:${LIBRARY_PATH:-}"

# Intel oneAPI repo
ONEAPI_VERSION=2026.0
tee /etc/yum.repos.d/oneapi.repo <<EOF
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

dnf install -y \
    intel-oneapi-dnnl-devel-${ONEAPI_VERSION} \
    intel-oneapi-mkl-devel-${ONEAPI_VERSION} \
    intel-oneapi-compiler-dpcpp-cpp-${ONEAPI_VERSION} \
    intel-oneapi-ccl-devel \
    intel-oneapi-mpi-devel

[ -f /opt/intel/oneapi/setvars.sh ] && source /opt/intel/oneapi/setvars.sh

if ! command -v icpx &> /dev/null; then
    for dir in /opt/intel/oneapi/compiler/*/bin; do
        [ -x "$dir/icx" ] && export PATH="$dir:$PATH" && break
    done
fi

command -v icpx &> /dev/null || {
    echo "FATAL: icpx not found. WITH_XPU=ON requires DPC++ compiler."
    exit 1
}

C_COMPILER=icx
CXX_COMPILER=icpx

mkdir build-release && cd build-release

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=${C_COMPILER} \
      -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
      -DCMAKE_CXX_FLAGS="-msse4.1" \
      -DBUILD_CLI=OFF \
      -DWITH_MKL=ON \
      -DWITH_XPU=ON \
      -DWITH_TENSOR_PARALLEL=ON \
      -DOPENMP_RUNTIME=INTEL \
      ..

VERBOSE=1 make -j$(nproc) install
cd ..
rm -r build-release

cp README.md python/
