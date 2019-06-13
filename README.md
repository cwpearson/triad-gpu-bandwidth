# Triad GPU Bandwidth

Test achievable triad kernel bandwidth using all available CUDA allocation/transfer methodologies:
* pageable allocation and explicit transfer: `cudaMalloc` / `cudaMemcpy`
* pinned allocatiion and explicit transfer: `cudaHostAlloc` / `cudaMemcpy`
* pinned allocation with mapping: `cudaHostAlloc(..., cudaHostAllocMapped)`
* unified memory: `cudaMallocManaged`
* unified memory with `AccessedBy` hint: `cudaMallocManaged + cudaMemAdvise(..., cudaMemAdviseAccessedBy)`
* unified memory with prefetching: `cudaMallocManaged + cudaMemPrefetchAsync`
* system allocator: `new`

The basic operation is a host-to-device transfer, followed by the triad kernel, followed by a device-to-host transfer.

The output is a CSV file with columns for the transfer time (zero for implicit transfers), the kernel time, and the total time.

## Automatic Testing for Functional System Allocator
* Forks a child process to test whether CUDA can use the system allocator.
If CUDA cannot, this causes a sticky error that permanently damages the CUDA context, so we use a child process to fully isolate the context so it can be completely destroyed.
The child process needs to be create before the parent does *any* CUDA activity at all.

The test is done in [test_system_allocator.cu](test_system_allocator.cu)

## Building on Power9 with gcc 4.8.5

GCC 4.8.5 doesn't have a working std::regex (used for cxxopts), so install a supported version of clang

|CUDA | Clang | Installer |
|-|-|-|
| 9.2  | 5.0.0 | https://gist.github.com/cwpearson/c5521dfc50175b1d977643b2fc5a2bb1 |
| 10.1 | 8.0.0 | https://gist.github.com/cwpearson/fc91b92c3d49d75a1b3a559aacb1d38e |

Then, build and tell CMake to use clang for everything

```
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=`readlink -f ../toolchains/clang.toolchain`
```

## Building with a gcc that has std::regex

```
mkdir build && cd build
cmake ..
```
