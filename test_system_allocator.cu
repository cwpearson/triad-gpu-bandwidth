#include <cstdio>

#include <sys/wait.h>
#include <unistd.h>

#include "check_cuda.cuh"

// write a single int
__global__ void write_int(int *a) {
    if (0 == blockDim.x * blockIdx.x + threadIdx.x) {
      *a = 700;
    }
  }

bool test_system_allocator() {

  // test the system allocator in a new process
  pid_t pid = fork(); // create child process
  int status;
  switch (pid) {
  case -1: // error
    perror("fork");
    exit(1);


  /* Test the system allocator by trying a write to a system allocation
     If fails, exit with non-zero. Otherwise exit zero.
     The parent will check the child's exit status to determine if it worked.
  */
  case 0: // child process
  {
    int *a = new int;
    write_int<<<1,1>>>(a);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaErrorIllegalAddress) {
        fprintf(stderr, "got illegal address using system allocator\n");
        exit(1);
    }
    CUDA_RUNTIME(cudaDeviceReset());
    exit(0);
}

  default: // parent process, pid now contains the child pid
    while (-1 == waitpid(pid, &status, 0))
      ; // wait for child to complete
    if (WIFSIGNALED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "test process exited with (%d). disabling\n", status);
      return false;
    } else {
      return true;
    }
  }

}