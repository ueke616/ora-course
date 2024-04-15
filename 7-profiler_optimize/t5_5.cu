// compile with: nvcc -Xcompiler -fopenmp -o t5 t5.cu -O3 -lineinfo
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:               return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:       return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:          return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:         return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:         return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:         return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:      return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "";
}


#include <sys/time.h>
#define USECPSEC 1000000ULL

// 计时函数，用于计算时间差，单位为微秒
unsigned long long dtime_usec(unsigned long long start){
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

// perform vector averaging over M vectors of length L,  followed by matrix-vector multiply
// repeat the above N times
// input vectors are stored as a set of N column-major matrices
// for each k in N: output[k] = matrix*input[k]
// cpu 版本的向量平均和矩阵-向量乘法
template<typename T>
void cpu_version1(T *input, T *output, T *matrix, int L, int M, int N){
#pragma omp parallel for
  for (int k = 0; k < N; k++){      // repeat the following, N times
    std::vector<T> v1(L);           // vector length of L
    for (int i = 0; i < M; i++)     // compute average vector over M input vectors
      for (int j = 0; j < L; j++)
        v1[j] += input[k*M*L+j*M+i];
    for (int j = 0; j < L; j++)
      v1[j] /= M;
    for (int i = 0; i < L; i++)     // matrix-vector multiply
      for (int j = 0; j < L; j++)
        output[i*N+k] += matrix[i*L+j]*v1[j];
  }
}

const int my_L = 1024;
const int my_M = 1024;
const int my_N = 1024;

template<typename T>
__global__ void gpu_version5(const T* __restrict__ input, T* __restrict__ output, const int L, const int M, const int N) {
  // parallelize threadIdx.x over vector length, and blockIdx.x across k(N)
  // do initial vector reduction via warp-stride loop
  int k = blockIdx.x;
  T v1 = 0;
  for (int y = threadIdx.y; y < L; y += blockDim.y) { // vertical block-stride loop
    v1 = 0;
    for (int x = threadIdx.x; x<M; x+=warpSize)
      v1 += input[k*M*L + y*M +x];
      for (int offset = warpSize>>1; offset > 0; offset >>= 1) // warp-shuffle reduction
        v1 += __shfl_down_sync(0xFFFFFFFF, v1, offset);
    if (!threadIdx.x)
      output[k+y*N] = v1/M;
  }
}

typedef float ft;

int main(){
  ft *d_input, *h_input, *d_output, *h_outputc, *h_outputg, *d_matrix, *h_matrix, *d_result;
  int L = my_L; int M = my_M; int N = my_N;
  // host allocations
  h_input   = new ft[N*L*M];
  h_matrix  = new ft[L*L];
  h_outputg = new ft[N*L];
  h_outputc = new ft[N*L];
  // data initialization
  for (int i = 0; i < N*L*M; i++) h_input[i] = (rand()&1)+1;  // 1 or 2
  for (int i = 0; i < L*L; i++) h_matrix[i]  = (rand()&1)+1;  // 1 or 2
  // create result to test for correctness
  unsigned long long dt = dtime_usec(0);
  cpu_version1(h_input, h_outputc, h_matrix, L, M, N);
  dt = dtime_usec(dt);
  std::cout << "CPU execution time: " << dt/(float)USECPSEC << "s" << std::endl;
  // device allocations
  cudaMalloc(&d_input, N*L*M*sizeof(ft));
  cudaMalloc(&d_output,  N*L*sizeof(ft));
  cudaMalloc(&d_matrix,  L*L*sizeof(ft));
  cudaMalloc(&d_result, N*L*sizeof(ft));
  cudaCheckErrors("cudaMalloc failure");
  // copy input data from host to device
  cudaMemcpy(d_input,  h_input,  N*L*M*sizeof(ft), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, h_matrix,   L*L*sizeof(ft), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, N*L*sizeof(ft));
  cudaCheckErrors("cudaMemcpy/Memset failure");
  
  // cublas setup
  cublasHandle_t h;
  ft alpha = 1.0;
  ft beta  = 1.0;
  cublasStatus_t c_res = cublasCreate(&h);
  if (c_res != CUBLAS_STATUS_SUCCESS){
    std::cout << "CUBLAS create error: " << _cudaGetErrorEnum(c_res) << std::endl;
    return 0;
  }
  // run on device and measure execution time
  dim3 block(32, 32);
  dt = dtime_usec(0);
  gpu_version5<<<N, block>>>(d_input, d_output, L, M, N);
  cudaCheckErrors("kernel launch failure");
  c_res = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_T,
                    N, N, L, &alpha,
                    d_matrix, L,
                    d_output, N, &beta,
                    d_result, N
                  );
  // cublas 中，有一个比 cublasSgemm 更高效的版本，cublasGemmEx，它使用了 TensorCore，更加高效
  // c_res = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_T,
  //   N, N, L, &alpha,
  //   d_matrix, CUDA_R_32F, L,
  //   d_output, CUDA_R_32F, N, &beta,
  //   d_result, CUDA_R_32F, N,
  //   CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT
  // );

  if (c_res != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS gemm error: " << _cudaGetErrorEnum(c_res) << std::endl;
    return 0;
  }

  cudaDeviceSynchronize();
  cudaCheckErrors("execution failure");
  
  dt = dtime_usec(dt);
  cudaMemcpy(h_outputg, d_result, N*L*sizeof(ft), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy failure");

  for (int i=0; i<N; i++)
    for (int j = 0; j < L; j++) 
      if (h_outputg[i+N*j] != h_outputc[j+N*i]) {
        std::cout << "Mismatch at " << i << " was: " << h_outputg[i] << " should be: " << h_outputc[i] << std::endl; 
        return 0;
      }
  std::cout << "Kernel execution time: " << dt/(float)USECPSEC << "s" << std::endl;
  return 0;
}