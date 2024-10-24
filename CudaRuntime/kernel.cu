
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void add(int* a, int* b, int* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {  
        c[index] = a[index] + b[index];
    }
}

int* dev_a;
int* dev_b;
int* dev_c;
int size;
cudaStream_t stream;

extern "C" __declspec(dllexport) void create(int _size) {
    size = _size;  // 전역 변수에 사이즈 저장

    // GPU 메모리 할당
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // 스트림 생성
    cudaStreamCreate(&stream);
}

extern "C" __declspec(dllexport) void copyToDevice(int* a, int* b) {
    // 비동기 전송 (CPU -> GPU)
    cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream);
}
extern "C" __declspec(dllexport) void addArrays() {
    // 비동기 커널 실행
    int threadsPerBlock = 512;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    add <<< numBlocks, threadsPerBlock, 0, stream >>> (dev_a, dev_b, dev_c, size);
}

extern "C" __declspec(dllexport) void copyToHost(int* c) {
    // 비동기 전송 (GPU -> CPU)
    cudaMemcpyAsync(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost, stream);
}

extern "C" __declspec(dllexport) void release() {
    // 스트림 작업 완료 기다림
    cudaStreamSynchronize(stream);

    // 스트림과 메모리 해제
    cudaStreamDestroy(stream);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}