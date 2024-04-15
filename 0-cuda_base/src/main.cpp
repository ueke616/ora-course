#include "cuda_runtime.h"

__global__ void simplekernel(void){
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
}
