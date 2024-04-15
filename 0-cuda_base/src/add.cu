// nvcc add.cu -o add && ./add
#include "cuda_runtime.h"
#include <stdio.h>
#include <memory>

#define N (1024 * 1024)
#define THREADS_PER_BLOCK 512

class Obj {
public:
    Obj(int a): a_(a) {}
    Obj(const Obj &obj){
        puts("copy constructor called");
        this->a_ = obj.a_;
    }
    ~Obj(){}
    int a_;
};

// kernel 函数通常没有返回值，对它的错误检查需要用到宏定义
#define CHECK(call)        \
do{                        \
    const cudaError_t error_code = call; \
    if (error_code != cudaSuccess){      \
        printf("CUDA Error: \n");        \
        printf("    FILE:    %s\n", __FILE__);    \
        printf("    LINE:    %d\n", __LINE__);    \
        printf("    Error code: %d\n", error_code);  \
        printf("    Error text: %s\n", cudaGetErrorString(error_code));    \
        exit(1);                        \
    }    \
} while(0)

template<typename T>    // 模板化
__global__ void addkernel(T* a, T* b, T* c, int n, Obj obj) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    // 只打印一次, 不然终端太多信息
    if (!index) {
        printf("obj.a_ : %d\n", obj.a_);
        obj.a_ = 200;
    }
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main(){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    a = (int*)malloc(size);    for(int i=0; i<N; i++) a[i] = i;
    b = (int*)malloc(size);    for(int i=0; i<N; i++) b[i] = 2 * i;
    c = (int*)malloc(size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    // printf("cudaMemcpy function: %s\n", cudaMemcpyHostToDevice则会报错)
    // 如果使用 cudaMemcpyDefault, 交换 d_a 和 a 不会报错，cudaMemcpyHostToDevice 则会报错
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // 如果使用 cudaMemcpyDefault, 交换d_b 和 b 不会报错, cudaMemcpyHostToDevice 则会报错
    
    dim3 n{THREADS_PER_BLOCK, 1, 1}; // blockDim.x 和 gridDim.x 可以是运行期参数, 不一定在编译时就确定
    dim3 m{(N + n.x - 1)/n.x, 1, 1};  // a/b 上取整 -> (a+b-1)/b
    
    // std::unique_ptr<int> d_ap(d_a);
    // std::unique_ptr<int> d_bp(d_b);
    // std::unique_ptr<int> d_cp(d_c);
    // printf("d_ap.get(): %p, d_bp.get(): %p, d_cp.get(): %p\n", d_ap.get(), d_bp.get(), d_cp.get());
    // addkernel<<<m,n>>>(d_ap.get(), d_bp.get(), d_cp.get(), N);
    
    Obj obj(100);
    
    addkernel<<<m, n>>>(d_a, d_b, d_c, N, obj);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    for(int  i = 0; i < 5; i++) {
        printf("gpu result %d, cpu result %d \n", c[i], 3 * i);
    }
    
    printf("after addkernel called, obj._a : %d\n", obj.a_);
    
    free(a);
    free(b);
    free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}        