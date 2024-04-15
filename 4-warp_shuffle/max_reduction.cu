#include <stdio.h>
#include <float.h>

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        printf("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
    }} while(0);


const size_t N = 8ULL*1024ULL*1024ULL;  // data size
const int BLOCK_SIZE = 256;  // CUDA maximum is 1024

__global__ void reduce_traditional(float *gdata, float *out, size_t n) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    float max_val = -FLT_MAX;  // FLT_MAX 代表在 float 类型所能表示的最大正值

    // 所有线程共同执行规约操作
    for (size_t i = idx; i < n; i += stride) {
        max_val = max(gdata[i], max_val);
    }

    // 在全局内存中对每个 block 的结果进行合并
    atomicMax((int*)out, __float_as_int(max_val));
}

__global__ void reduce(float *gdata, float *out, size_t n){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < n) {  // grid stride loop to load data
        sdata[tid] = max(gdata[idx], sdata[tid]);
        idx += gridDim.x*blockDim.x;  
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // parallel sweep reduction
            sdata[tid] = max(sdata[tid + s], sdata[tid]);
        }
     if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main(){

  float *h_A, *h_sum, *d_A, *d_sums;
  
  const int blocks = 3072; // 4090 max blocks per gpu
  h_A = new float[N];  // allocate space for data in host memory
  h_sum = new float;
  float max_val = 5.0f;
  
  for (size_t i = 0; i < N; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  h_A[100] = max_val;
  
  checkRuntime(cudaMalloc(&d_A, N*sizeof(float)));  // allocate device space for A
  checkRuntime(cudaMalloc(&d_sums, blocks*sizeof(float)));  // allocate device space for partial sums

  // copy matrix A to device:
  checkRuntime(cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice));

  // sweep parallel reduction
  //cuda processing sequence step 1 is complete
  checkCudaKernel(reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N)); // reduce stage 1
  checkCudaKernel(reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks)); // reduce stage 2

  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  checkRuntime(cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost));
  //cuda processing sequence step 3 is complete
  printf("reduction output: %f, expected sum reduction output: %f, expected max reduction output: %f\n", *h_sum, (float)((N-1)+max_val), max_val);

  // //traditional reduction
  // // 第一阶段规约：每个block的规约结果写入全局内存
  // checkCudaKernel(reduce_traditional<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N));

  // // 第二阶段规约：在CPU上进行最终规约
  // float *h_sums = new float[blocks];
  // checkRuntime(cudaMemcpy(h_sums, d_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

  // float final_max = -FLT_MAX;
  // for(int i = 0; i < blocks; ++i) {
  //     final_max = max(h_sums[i], final_max);
  // }

  // //cuda processing sequence step 3 is complete
  // printf("reduction output: %f, expected sum reduction output: %f, expected max reduction output: %f\n", final_max, (float)((N-1)+max_val), max_val);

  cudaFree(d_A);
  cudaFree(d_sums);
  return 0;
}