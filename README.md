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

* total time: the time from the start of the first transfer to the end of the last transfer.
* transfer time: the combined time for the host-to-device and the device-to-host transfers.
* kernel time: start of the kernel execution to end of the kernel execution

**transfer + kernel may not equal total**, though they should be close.

## Examples

* Run benchmarks on GPU 0 from `n = 1e5; n <= 2.5e8; n *= 1.3`.
Repeat each benchmark 5 times.
Pin access and allocations to NUMA node 0.
Show output on terminal and also pipe to `triad.csv`.

`numactl -p 0 ./triad | tee triad.csv`

* Run only the `pinned` benchmark

`./triad --pinned`

* Run only n = 1e9

`./triad -n 1e9`

* Run 3 iterations of each benchmark

`./triad -i 3`

* Show all options

`./triad -h`

## Automatic Testing for Functional System Allocator
Forks a child process to test whether CUDA can use the system allocator.
If CUDA cannot, this causes a sticky error that permanently damages the CUDA context, so we use a child process to fully isolate the context so it can be completely destroyed.
The child process needs to be create before the parent does *any* CUDA activity at all.

The test is done in [test_system_allocator.cu](test_system_allocator.cu)

## Getting stable benchmark results

It is important to do 3 things:
1. Call `cudaDeviceReset()` before each benchmark. This ensures that any CUDA state is wiped between runs.
2. Call `cudaFree(0)` after `cudaDeviceReset()`. This initializes the GPU, ensuring that we don't actidentally time any lazy initialization.
3. Pin to a single NUMA region or CPU. This ensures that data copies always take a consistent route from CPU to GPU.

## Building with a gcc that has std::regex

```
mkdir build && cd build
cmake ..
```

## Building on Power9 with gcc 4.8.5

GCC 4.8.5 doesn't have a working std::regex (used for cxxopts), so install a supported version of clang.
GCC 4.8.5 cannot build libcxx, so we use a clang without libcxx to build a clang with libcxx.
Depending on your installed CUDA, you'll need a different version of clang.

| CUDA | Clang | Installer |
|-|-|-|
| 9.2  | 5.0.0 | https://gist.github.com/cwpearson/c5521dfc50175b1d977643b2fc5a2bb1 |
| 10.1 | 7.1.0 | https://gist.github.com/cwpearson/c13ac7c25bde8c8644300e211faf4e78 |

Add the clang to your path, and have CMake use clang in the build.

```
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=`readlink -f ../toolchains/clang.toolchain`
```
The CUDA documentation claims that clang 8.0.0 is supported for CUDA 10.1, but if you actually try it says it requires clang>=3.2 and clang<8.

