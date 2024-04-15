// compile with: nvcc -Xcompiler -fopenmp -o t5 t5.cu -O3 -lineinfo
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

// GPU 版本的向量平均和矩阵-向量乘法
template<typename T>
__global__ void gpu_version1(const T * __restrict__ input, T * __restrict__ output, const T * __restrict__ matrix, const int L, const int M, const int N){

  __shared__ T smem[my_L];
  size_t idx = ((size_t)blockIdx.x)*blockDim.x + threadIdx.x;
  for (int k = 0; k < N; k++){  // iterate over N data sets
    T v1 = 0;
    for (int i = 0; i < M; i++) // perform vector averaging
      v1 += input[k*M*L+idx*M+i];
    v1 /= M;
    for (int i = 0; i < L; i++){ // perform matrix-vector multiply
      __syncthreads();
      smem[threadIdx.x] = v1 * matrix[i*L+idx];
      for (int s = blockDim.x>>1; s > 0; s>>=1){
        __syncthreads();
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];}
      if (!threadIdx.x) output[k+i*N] = smem[0];}
  }
}

// gpu_version2 使得每个块专门处理一个数据集，而 gpu_version1 则要求线程跨数据集进行操作，这可能影响性能和效率，特别是在大规模数据处理时
template<typename T>
__global__ void gpu_version2(const T * __restrict__ input, T * __restrict__ output, const T * __restrict__ matrix, const int L, const int M, const int N){
    __shared__ T smem[my_L];
    size_t idx = threadIdx.x;
    size_t k = blockIdx.x;

    T v1 = 0;
    for (int i = 0; i < M; i++) // perform vector averaging
        v1 += input[k*M*L+idx*M+i];
        v1 /= M;
    for (int i = 0; i < L; i++){ // perform matrix-vector multiply
        __syncthreads();
        smem[threadIdx.x] = v1 * matrix[i*L+idx];
        for (int s = blockDim.x>>1; s > 0; s>>=1){
            __syncthreads();
            if (threadIdx.x < s) 
                smem[threadIdx.x] += smem[threadIdx.x+s];
        }
        if (!threadIdx.x) 
            output[k+i*N] = smem[0];
    }
}

typedef float ft;

int main(){
  ft *d_input, *h_input, *d_output, *h_outputc, *h_outputg, *d_matrix, *h_matrix;
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
  cudaCheckErrors("cudaMalloc failure");
  // copy input data from host to device
  cudaMemcpy(d_input,  h_input,  N*L*M*sizeof(ft), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, h_matrix,   L*L*sizeof(ft), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, N*L*sizeof(ft));
  cudaCheckErrors("cudaMemcpy/Memset failure");
  // run on device and measure execution time
  dt = dtime_usec(0);
  gpu_version2<<<N, L>>>(d_input, d_output, d_matrix, L, M, N);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel execution failure");
  dt = dtime_usec(dt);
  cudaMemcpy(h_outputg, d_output, N*L*sizeof(ft), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy failure");
  for (int i = 0; i < N*L; i++) if (h_outputg[i] != h_outputc[i]) {std::cout << "Mismatch at " << i << " was: " << h_outputg[i] << " should be: " << h_outputc[i] << std::endl; return 0;}
  std::cout << "Kernel execution time: " << dt/(float)USECPSEC << "s" << std::endl;
  return 0;
}