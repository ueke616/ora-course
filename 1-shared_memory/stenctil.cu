#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 1024      // (使用的)block数量
#define r 3         // 左右填补
#define BLOCK_SIZE 128      // 每个block的大小

__global__ void stencil_1d(int *in, int *out, int n){
    __shared__ int temp[BLOCK_SIZE + 2 *r];  // per-block, static allocation
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; //用来索引 global memory
    if (gindex >= n) return;    // 越界
    int lindex = threadIdx.x + r;  // 用来索引 shared memory
    
    // 把数据从 global memory 读到 shared memory 里。没有for循环, SIMT 模式，一个 block 里所有线程完成合作 从global memory 读数据到 shared memory 这个操作。
    // SIMT: Single Instruction Multiple Threads 单指令多线程
    temp[lindex] = in[gindex];
    if (threadIdx.x < r) {  // 考虑左右边界
        temp[lindex - r] = in[gindex - r];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
    
    // 读取之前需要同步一下，确保所有的写入任务都已完成
    __syncthreads();    // 确保所有数据已经读到 shared memory 里, 注意不能写在 if 条件语句里
    int res = 0;
    // 此时需要 for 循环, 因为一次线程要负责计算 2*r+1 个元素的和
    for (int offset = -r; offset <= r; offset++) {
        res += temp[lindex + offset];
    } 
    out[gindex] = res;
}

void fill_ints(int *x, int n) {
    fill_n(x, n, 1);
}

int main(void) {
    int *in, *out;
    int *d_in, *d_out;
    int size = (N * 2*r) * sizeof(int);

    // 在内存分配空间
    in      = (int *)malloc(size); fill_ints(in, N + 2*r);
    out     = (int *)malloc(size); fill_ints(out, N + 2*r);

    // 在显存分配空间
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // 数据从cpu拷贝到显存
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    // 运行核函数
    stencil_1d<<<(N + BLOCK_SIZE - 1 / BLOCK_SIZE), BLOCK_SIZE>>>(d_in + r, d_out + r, N);
    
    // 数据拷回cpu
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // 校对结果
    for (int i = 0; i < N + 2*r; i++) {
        if (i < r || i >= N + r) {  // 处理越界问题
            if (out[i] != 1)
                printf("Mismatch at index %d, was: %d., should be: %d\n", i, out[i], 1);
        } else {
            if (out[i] != 1 + 2*r)
                printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], i + 2*r);
        }

    }

    free(in); free(out);
    cudaFree(d_in); cudaFree(d_out);
    printf("Success!\n");
    return 0;
}
