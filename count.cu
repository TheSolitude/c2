#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h> // for sleep()

__global__ void increment_counter(int* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(counter, 1); // Safe for multi-threading (optional here)
    }
}

int main() {
    int* d_counter;
    int h_counter = 0;

    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

    printf("Starting GPU counter...\n");

    for (int i = 0; i < 10; i++) {  // run for 10 seconds
        increment_counter<<<1, 1>>>(d_counter);
        cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        printf("GPU Counter: %d\n", h_counter);
        sleep(1); // wait 1 second
    }

    cudaFree(d_counter);
    return 0;
}
