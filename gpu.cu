#include <stdio.h>
#include <cuda_runtime.h>

// CUDA core count per SM (Streaming Multiprocessor) for known architectures
int convertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int cores;
    } sSMtoCores;

    sSMtoCores archCores[] = {
        {0x30, 192}, // Kepler
        {0x32, 192}, // Kepler
        {0x35, 192}, // Kepler
        {0x37, 192}, // Kepler
        {0x50, 128}, // Maxwell
        {0x52, 128}, // Maxwell
        {0x53, 128}, // Maxwell
        {0x60,  64}, // Pascal
        {0x61, 128}, // Pascal
        {0x62, 128}, // Pascal
        {0x70,  64}, // Volta
        {0x72,  64}, // Xavier
        {0x75,  64}, // Turing
        {0x80,  64}, // Ampere
        {0x86, 128}, // Ampere (RTX 30 series)
        {0x89, 128}, // Ada Lovelace (RTX 40 series - L4)
        {0x90, 128}, // Hopper
        {-1, -1}
    };

    int index = 0;
    while (archCores[index].SM != -1) {
        if (archCores[index].SM == ((major << 4) + minor))
            return archCores[index].cores;
        index++;
    }

    // Default fallback
    printf("Unknown SM version %d.%d - assuming 64 cores/SM\n", major, minor);
    return 64;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int coresPerSM = convertSMVer2Cores(prop.major, prop.minor);
        int totalCores = coresPerSM * prop.multiProcessorCount;

        printf("=== Device %d ===\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("SM Count: %d\n", prop.multiProcessorCount);
        printf("Cores per SM: %d\n", coresPerSM);
        printf("Total CUDA Cores: %d\n", totalCores);
        printf("\n");
    }

    return 0;
}

