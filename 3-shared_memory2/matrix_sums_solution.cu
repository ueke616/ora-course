#include <stdio.h>

// 错误检查宏
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "致命错误：%s (%s 在 %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** 失败 - 中止程序\n"); \
            exit(1); \
        } \
    } while (0)

// 矩阵边的尺寸
const size_t DSIZE = 16384;
// CUDA的最大线程块尺寸
const int block_size = 256;

// 矩阵行求和核函数
__global__ void row_sums(const float *A, float *sums, size_t ds){
  // 根据内置变量创建典型的一维线程索引
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < ds){
    float sum = 0.0f;
    // 循环迭代行，计算运行和，并将结果写入sums
    for (size_t i = 0; i < ds; i++)
      sum += A[idx*ds+i];
    sums[idx] = sum;
}}

// 矩阵列求和核函数
__global__ void column_sums(const float *A, float *sums, size_t ds){
  // 根据内置变量创建典型的一维线程索引
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < ds){
    float sum = 0.0f;
    // 循环迭代列，计算运行和，并将结果写入sums
    for (size_t i = 0; i < ds; i++)
      sum += A[idx+ds*i];
    sums[idx] = sum;
}}

// 验证函数
bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {
      printf("结果不匹配在 %lu, 结果是: %f, 应该是: %f\n", i, data[i], (float)sz);
      return false;
    }
    return true;
}

int main(){
    float *h_A, *h_sums, *d_A, *d_sums;
    
    // 在主机内存中为数据分配空间
    h_A = new float[DSIZE*DSIZE];  
    h_sums = new float[DSIZE]();
      
    // 初始化主机内存中的矩阵
    for (int i = 0; i < DSIZE*DSIZE; i++)  
      h_A[i] = 1.0f;
      
    // 在设备上为矩阵A分配空间
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  
    // 为d_sums向量在设备上分配空间
    cudaMalloc(&d_sums, DSIZE*sizeof(float));  
    // 错误检查
    cudaCheckErrors("cudaMalloc分配失败"); 
      
    // 将矩阵A复制到设备上
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    // 错误检查
    cudaCheckErrors("cudaMemcpy主机到设备复制失败");
    // CUDA处理序列的第一步完成
      
    // 调用核函数进行行求和
    row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    // 错误检查
    cudaCheckErrors("核函数启动失败");
    // CUDA处理序列的第二步完成
      
    // 将求和结果的向量从设备复制回主机
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    // CUDA处理序列的第三步完成
      
    // 错误检查
    cudaCheckErrors("核函数执行失败或cudaMemcpy设备到主机复制失败");
    // 验证求和结果是否正确
    if (!validate(h_sums, DSIZE)) return -1; 
    printf("行求和正确！\n");
      
    // 将设备上的d_sums向量置零，以便再次使用
    cudaMemset(d_sums, 0, DSIZE*sizeof(float));
      
    // 调用核函数进行列求和
    column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    // 错误检查
    cudaCheckErrors("核函数启动失败");
    // CUDA处理序列的第二步完成
      
    // 将求和结果的向量从设备复制回主机
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    // CUDA处理序列的第三步完成
      
    // 错误检查
    cudaCheckErrors("核函数执行失败或cudaMemcpy设备到主机复制失败");
    // 验证求和结果是否正确
    if (!validate(h_sums, DSIZE)) return -1; 
    printf("列求和正确！\n");
    // 主函数结束，返回0表示成功
    return 0;
  }
  
  