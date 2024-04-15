// 演示block sheduler 的示例代码
// 一个SM里能同时驻留的block数量是有限制的，（Maxwell/Pascal/Volta:32）
// ./block_scheduler 1
#include <cstdio>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) tp.tv_sec + (double) tp.tv_usec * 1e-6;
}

#define CLOCK_RATE 1695000 /*Modify from below*/
__device__ void sleep(float t) {
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0)/(CLOCK_RATE*1000.0f) < t) {
        t1 = clock64();
    }
}

__global__ void mykernel() {
    sleep(1.0);
}

int main(int argc, char* argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int mp = prop.multiProcessorCount;  // 64
    int mbp = prop.maxBlocksPerMultiProcessor;  // 16
    printf("*num of multiprocessor  %10d\n", mp);
    printf("*max blocks per multiprocessor  %10d\n", mbp);
    printf("*max blocks per gpu  %10d \n", mbp * mp);  // GPU上能同时驻留的Block的数量
    // clock_t clock_rate = prop.clockRate;
    // printf("clock_rate   %10d\n", clock_rate);
    int num_blocks = atoi(argv[1]);
    
    dim3 block(1);
    dim3 grid(num_blocks);  /* N blocks */
    // 64 * 16 = 1024, 2048, 3072, 4096
    
    double start = cpuSecond();
    mykernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    double etime = cpuSecond() - start;
    
    printf("mp        %10d\n", mp);
    printf("blocks/SM %10.2f\n", num_blocks/((double)mp));
    printf("time      %10.2f\n", etime);
    
    cudaDeviceReset();
}