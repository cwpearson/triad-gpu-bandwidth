#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "cxxopts.hpp"

#include "cache.hpp"
#include "check_cuda.cuh"

struct Result {
  double kernel;
  double copy;
  double total;
};

template <typename T>
__global__ void triad_kernel(T *__restrict__ a, const T *__restrict__ b,
                             const T *__restrict__ c, const T scalar,
                             const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}



// write a single int
__global__ void write_int(int *a) {
  if (0 == blockDim.x * blockIdx.x + threadIdx.x) {
    *a = 700;
  }
}

bool try_system_allocator() {
  int *a = new int;
  write_int<<<1,1>>>(a);
  cudaError_t err = cudaDeviceSynchronize();
  delete a;
  if (err == cudaErrorIllegalAddress) {
    fprintf(stderr, "got illegal address using system allocator\n");
    return false;
  }
  CUDA_RUNTIME(cudaDeviceReset());
  return true;
}

typedef enum {
  PAGEABLE,
  PINNED,
  ZERO_COPY,
  MANAGED,
  SYSTEM,
} AllocationType;

typedef enum {
  NONE = 0x0,
  ACCESS = 0x1,
  PREFETCH = 0x2,
} Hint;

inline Hint operator|(Hint a, Hint b) {
  return static_cast<Hint>(static_cast<int>(a) | static_cast<int>(b));
}

template <typename T>
Result benchmark_naive(size_t n, AllocationType at, Hint hint) {

  T *a_h = nullptr;
  T *b_h = nullptr;
  T *c_h = nullptr;

  switch (at) {
  case PAGEABLE:
  case SYSTEM:
    a_h = new T[n];
    b_h = new T[n];
    c_h = new T[n];
    break;
  case PINNED:
    CUDA_RUNTIME(cudaHostAlloc(&a_h, n * sizeof(T), 0));
    CUDA_RUNTIME(cudaHostAlloc(&b_h, n * sizeof(T), 0));
    CUDA_RUNTIME(cudaHostAlloc(&c_h, n * sizeof(T), 0));
    break;
  case ZERO_COPY:
    CUDA_RUNTIME(cudaHostAlloc(&a_h, n * sizeof(T), cudaHostAllocMapped));
    CUDA_RUNTIME(cudaHostAlloc(&b_h, n * sizeof(T), cudaHostAllocMapped));
    CUDA_RUNTIME(cudaHostAlloc(&c_h, n * sizeof(T), cudaHostAllocMapped));
    break;
  case MANAGED:
    CUDA_RUNTIME(cudaMallocManaged(&a_h, n * sizeof(T)));
    CUDA_RUNTIME(cudaMallocManaged(&b_h, n * sizeof(T)));
    CUDA_RUNTIME(cudaMallocManaged(&c_h, n * sizeof(T)));
    break;
  default:
    fprintf(stderr, "unexpected AllocationType\n");
    exit(1);
  }

  // touch all pages
  // fprintf(stderr, "touch all pages\n");
  for (size_t i = 0; i < n; i += 32) {
    a_h[i] = i;
    b_h[i] = i;
    c_h[i] = i;
  }

  // fprintf(stderr, "init dev pointers\n");
  T *a_d = nullptr;
  T *b_d = nullptr;
  T *c_d = nullptr;

  switch (at) {
  case PAGEABLE:
  case PINNED:
    CUDA_RUNTIME(cudaMalloc(&a_d, sizeof(T) * n));
    CUDA_RUNTIME(cudaMalloc(&b_d, sizeof(T) * n));
    CUDA_RUNTIME(cudaMalloc(&c_d, sizeof(T) * n));
    break;
  case ZERO_COPY:
    CUDA_RUNTIME(cudaHostGetDevicePointer(&a_d, a_h, 0));
    CUDA_RUNTIME(cudaHostGetDevicePointer(&b_d, c_h, 0));
    CUDA_RUNTIME(cudaHostGetDevicePointer(&c_d, c_h, 0));
    break;
  case MANAGED:
  case SYSTEM:
    a_d = a_h;
    b_d = b_h;
    c_d = c_h;
    break;
  }

  // fprintf(stderr, "create events\n");
  cudaEvent_t kernelStart, kernelStop;
  cudaEvent_t txStart, txStop;
  cudaEvent_t rxStart, rxStop;
  CUDA_RUNTIME(cudaEventCreate(&kernelStart));
  CUDA_RUNTIME(cudaEventCreate(&kernelStop));
  CUDA_RUNTIME(cudaEventCreate(&txStart));
  CUDA_RUNTIME(cudaEventCreate(&txStop));
  CUDA_RUNTIME(cudaEventCreate(&rxStart));
  CUDA_RUNTIME(cudaEventCreate(&rxStop));

  // flush caches
  // fprintf(stderr, "flush\n");
  flush_all(a_h, sizeof(T) * n);
  flush_all(b_h, sizeof(T) * n);
  flush_all(c_h, sizeof(T) * n);

  // fprintf(stderr, "h2d\n");
  CUDA_RUNTIME(cudaEventRecord(txStart));
  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaMemcpyAsync(a_d, a_h, sizeof(T) * n, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpyAsync(b_d, b_h, sizeof(T) * n, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpyAsync(c_d, c_h, sizeof(T) * n, cudaMemcpyDefault));
  }
  if ((at == MANAGED) && (PREFETCH & hint)) {
    CUDA_RUNTIME(cudaMemPrefetchAsync(a_d, sizeof(T) * n, 0));
    CUDA_RUNTIME(cudaMemPrefetchAsync(b_d, sizeof(T) * n, 0));
    CUDA_RUNTIME(cudaMemPrefetchAsync(c_d, sizeof(T) * n, 0));
  }
  if ((at == MANAGED) && (ACCESS & hint)) {
    CUDA_RUNTIME(
        cudaMemAdvise(a_d, sizeof(T) * n, cudaMemAdviseSetAccessedBy, 0));
    CUDA_RUNTIME(
        cudaMemAdvise(b_d, sizeof(T) * n, cudaMemAdviseSetAccessedBy, 0));
    CUDA_RUNTIME(
        cudaMemAdvise(c_d, sizeof(T) * n, cudaMemAdviseSetAccessedBy, 0));
  }
  CUDA_RUNTIME(cudaEventRecord(txStop));

  int dimBlock = 512;
  int dimGrid = (n + dimBlock - 1) / dimBlock;

  // fprintf(stderr, "launch\n");
  CUDA_RUNTIME(cudaEventRecord(kernelStart));
  triad_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, 1, n);
  CUDA_RUNTIME(cudaEventRecord(kernelStop));

  // fprintf(stderr, "d2h\n");
  CUDA_RUNTIME(cudaEventRecord(rxStart));
  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaMemcpyAsync(c_h, c_d, sizeof(T) * n, cudaMemcpyDefault));
  }
  CUDA_RUNTIME(cudaEventRecord(rxStop));

  // fprintf(stderr, "times\n");
  CUDA_RUNTIME(cudaDeviceSynchronize());
  float txMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&txMillis, txStart, txStop));
  float rxMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&rxMillis, rxStart, rxStop));
  float kernelMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&kernelMillis, kernelStart, kernelStop));
  float totalMillis;
  CUDA_RUNTIME(cudaEventElapsedTime(&totalMillis, txStart, rxStop));

  // fprintf(stderr, "cuda free\n");
  if (at == PAGEABLE || at == PINNED) {
    CUDA_RUNTIME(cudaFree(a_d));
    CUDA_RUNTIME(cudaFree(b_d));
    CUDA_RUNTIME(cudaFree(c_d));
  }

  // fprintf(stderr, "host free\n");
  switch (at) {
  case PAGEABLE:
  case SYSTEM:
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
    break;
  case PINNED:
  case ZERO_COPY:
    CUDA_RUNTIME(cudaFreeHost(a_h));
    CUDA_RUNTIME(cudaFreeHost(b_h));
    CUDA_RUNTIME(cudaFreeHost(c_h));
    break;
  case MANAGED:
    CUDA_RUNTIME(cudaFree(a_h));
    CUDA_RUNTIME(cudaFree(b_h));
    CUDA_RUNTIME(cudaFree(c_h));
    break;
  default:
    fprintf(stderr, "unexpected AllocationType\n");
    exit(1);
  }

  a_h = nullptr;
  b_h = nullptr;
  c_h = nullptr;

  // fprintf(stderr, "destroy event\n");
  CUDA_RUNTIME(cudaEventDestroy(kernelStart));
  CUDA_RUNTIME(cudaEventDestroy(kernelStop));
  CUDA_RUNTIME(cudaEventDestroy(txStart));
  CUDA_RUNTIME(cudaEventDestroy(txStop));
  CUDA_RUNTIME(cudaEventDestroy(rxStart));
  CUDA_RUNTIME(cudaEventDestroy(rxStop));

  double copyPerf =
      1000.0 * n * sizeof(T) * 3 / (txMillis + rxMillis) / 1024 / 1024;
  double kernelPerf = 1000.0 * n * sizeof(T) * 3 / kernelMillis / 1024 / 1024;
  double totalPerf = 1000.0 * n * sizeof(T) * 3 / totalMillis / 1024 / 1024;

  // no copies in some of these
  if (at == ZERO_COPY) {
    copyPerf = -1;
  }
  if ((at == MANAGED) && (hint == NONE)) {
    copyPerf = -1;
  }
  if ((at == SYSTEM)) {
    copyPerf = -1;
  }

  Result result;
  result.kernel = kernelPerf;
  result.copy = copyPerf;
  result.total = totalPerf;
  // printf("%f.2 %f.2 %f.2\n", copyPerf, kernelPerf, totalPerf);
  return result;
}

void print_results(const std::vector<Result> results, const std::string &sep) {

  for (auto &result : results) {
    printf("%s", sep.c_str());
    if (result.copy >= 0) {
      printf("%.2e", result.copy);
    }
  }
  for (auto &result : results) {
    printf("%s", sep.c_str());
    if (result.kernel >= 0) {
      printf("%.2e", result.kernel);
    }
  }
  for (auto &result : results) {
    printf("%s", sep.c_str());
    if (result.total >= 0) {
      printf("%.2e", result.total);
    }
  }
  std::cout << std::endl;
}

template <typename T> std::vector<Result> run_many(size_t iters, T fn) {
  std::vector<Result> results;
  for (size_t i = 0; i < iters; ++i) {
    auto result = fn();
    results.push_back(result);
  }
  return results;
}

int main(int argc, char **argv) {

  bool disableSystemAllocator;

  // test the system allocator in a new process
  pid_t pid = fork(); // create child process
  int status;
  switch (pid)
  {
  case -1: // error
      perror("fork");
      exit(1);

  case 0: // child process 
      if (try_system_allocator()) {
        exit(0);
      }
      exit(1);

  default: // parent process, pid now contains the child pid
      while (-1 == waitpid(pid, &status, 0)); // wait for child to complete
      if (WIFSIGNALED(status) || WEXITSTATUS(status) != 0)
      {
          fprintf(stderr,"system allocator did not work (%d). disabling\n", status);
          disableSystemAllocator = true;
      } else {
        disableSystemAllocator = false;
      }
      break;
  }


  CUDA_RUNTIME(cudaDeviceReset());

  std::string sep = ",";
  size_t iters = 5;

  cxxopts::Options options("triad", "triad benchmarks");

  std::vector<double> gs;
  std::vector<double> ms;

  options.add_options()("n,num-iters", "Number of iterations",
                        cxxopts::value(iters))(
      "no-system-allocator", "Disable system allocator")("h,help", "Show help");

  auto result = options.parse(argc, argv);

  const bool help = result["help"].as<bool>();
  if (help) {
    printf("%s\n", options.help().c_str());
    exit(0);
  }

  if (result["no-system-allocator"].count()) {
    disableSystemAllocator = true;
  }


  // print header
  std::cout << "n" << sep << "bmark";
  for (size_t i = 0; i < iters; ++i) {
    std::cout << sep << "copy_" + std::to_string(i);
  }
  for (size_t i = 0; i < iters; ++i) {
    std::cout << sep << "kernel_" + std::to_string(i);
  }
  for (size_t i = 0; i < iters; ++i) {
    std::cout << sep << "total_" + std::to_string(i);
  }
  std::cout << std::endl;

  // runs
  // 3GB => 1GB each => n=250M
  for (size_t n = 1e5; n <= 2.5e8; n *= 1.3) {

    std::vector<Result> results;
    if (!disableSystemAllocator) {
      results =
          run_many(iters, std::bind(benchmark_naive<int>, n, SYSTEM, NONE));
      printf("%.2e%s%s", (double)n, sep.c_str(), "system          ");
      print_results(results, sep);
    }

    results =
        run_many(iters, std::bind(benchmark_naive<int>, n, PAGEABLE, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "pageable          ");
    print_results(results, sep);

    results = run_many(iters, std::bind(benchmark_naive<int>, n, PINNED, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "pinned            ");
    print_results(results, sep);

    results =
        run_many(iters, std::bind(benchmark_naive<int>, n, ZERO_COPY, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "zero-copy         ");
    print_results(results, sep);

    results =
        run_many(iters, std::bind(benchmark_naive<int>, n, MANAGED, NONE));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um                ");
    print_results(results, sep);

    results =
        run_many(iters, std::bind(benchmark_naive<int>, n, MANAGED, ACCESS));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um-access         ");
    print_results(results, sep);

    results =
        run_many(iters, std::bind(benchmark_naive<int>, n, MANAGED, PREFETCH));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um-prefetch       ");
    print_results(results, sep);

    results = run_many(
        iters, std::bind(benchmark_naive<int>, n, MANAGED, ACCESS | PREFETCH));
    printf("%.2e%s%s", (double)n, sep.c_str(), "um-access-prefetch");
    print_results(results, sep);
  }
}