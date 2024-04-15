#include <iostream>
#include <cuda_runtime.h>

__global__ void kernelRemoveSpacing(const int* inputArray, int* outputArray, int arraySize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= arraySize) return;
    if (inputArray[tid] != 0) {
        int pos = atomicAdd(outputArray + arraySize, 1);
        outputArray[pos] = inputArray[tid];
    }
}

int main() {
    const int arraySize = 10;
    int h_inputArray[arraySize] = {1, 0, 2, 0, 0, 3, 4, 0, 5, 0};
    int* d_inputArray;
    int* d_outputArray;

    std::cout << "Input Array Size: " << arraySize << std::endl;
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_inputArray[i] << " ";
    }
    std::cout << std::endl;

    cudaMalloc(&d_inputArray, arraySize * sizeof(int));
    cudaMalloc(&d_outputArray, (arraySize + 1) * sizeof(int)); // 额外一个空间用于计数
    cudaMemset(d_outputArray, 0, (arraySize + 1) * sizeof(int));

    cudaMemcpy(d_inputArray, h_inputArray, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    kernelRemoveSpacing<<<numBlocks, blockSize>>>(d_inputArray, d_outputArray, arraySize);

    int h_outputArray[arraySize + 1]; // 额外的空间用于存储输出数组的大小
    cudaMemcpy(h_outputArray, d_outputArray, (arraySize + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    int outputSize = h_outputArray[arraySize];
    std::cout << "Output Array Size: " << outputSize << std::endl;
    for (int i = 0; i < outputSize; i++) {
        std::cout << h_outputArray[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_inputArray);
    cudaFree(d_outputArray);

    return 0;
}
