// cuda_substitution_solver.cu
// Fixed version: atomicMax for float + host/device function + uppercase quadgrams

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#include <float.h>
#include <ctype.h>

#define NUM_RESTARTS 16384
#define THREADS_PER_BLOCK 256
#define MAX_ITERATIONS 20000
#define QUADGRAM_SIZE 456976
#define FLOOR_VALUE -12.0f

__constant__ float d_quadgrams[QUADGRAM_SIZE];
__constant__ char d_ciphertext[10000];
__constant__ int d_cipher_len;

// Make this callable from both host and device
__host__ __device__ int get_quadgram_index(int a, int b, int c, int d) {
    return ((a * 26 + b) * 26 + c) * 26 + d;
}

// Custom atomicMax for float (using atomicCAS)
__device__ float atomicMaxFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;

    do {
        assumed = old;
        float current = __uint_as_float(old);
        if (current >= val) return current;  // Early exit if already larger
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);

    return __uint_as_float(old);
}

__device__ float compute_fitness(const char* key) {
    float score = 0.0f;
    for (int i = 0; i < d_cipher_len - 3; ++i) {
        int c1 = d_ciphertext[i] - 'a';
        int c2 = d_ciphertext[i+1] - 'a';
        int c3 = d_ciphertext[i+2] - 'a';
        int c4 = d_ciphertext[i+3] - 'a';

        char p1 = key[c1];
        char p2 = key[c2];
        char p3 = key[c3];
        char p4 = key[c4];

        int idx = get_quadgram_index(p1 - 'a', p2 - 'a', p3 - 'a', p4 - 'a');
        score += d_quadgrams[idx];
    }
    return score;
}

__global__ void hill_climbing_kernel(float* best_global_score, char* best_global_key) {
    __shared__ float s_best_score;
    __shared__ char s_best_key[26];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(tid + clock64(), 0, 0, &state);

    char key[26];
    char best_key[26];
    for (int i = 0; i < 26; ++i) key[i] = 'a' + i;

    // Randomize initial key
    for (int i = 0; i < 1000; ++i) {
        int a = curand(&state) % 26;
        int b = curand(&state) % 26;
        if (a != b) {
            char temp = key[a];
            key[a] = key[b];
            key[b] = temp;
        }
    }

    float current_score = compute_fitness(key);
    float local_best_score = current_score;
    memcpy(best_key, key, 26);

    // Hill climbing
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        int a = curand(&state) % 26;
        int b = curand(&state) % 26;
        while (a == b) b = curand(&state) % 26;

        char temp = key[a];
        key[a] = key[b];
        key[b] = temp;

        float new_score = compute_fitness(key);

        if (new_score > current_score) {
            current_score = new_score;
            if (new_score > local_best_score) {
                local_best_score = new_score;
                memcpy(best_key, key, 26);
            }
        } else {
            temp = key[a];
            key[a] = key[b];
            key[b] = temp;
        }
    }

    // Block-level best
    if (threadIdx.x == 0) {
        s_best_score = -FLT_MAX;
    }
    __syncthreads();

    atomicMaxFloat(&s_best_score, local_best_score);

    __syncthreads();

    if (local_best_score == s_best_score) {
        memcpy(s_best_key, best_key, 26);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMaxFloat(best_global_score, s_best_score);
        if (s_best_score == *best_global_score) {
            memcpy(best_global_key, s_best_key, 26);
        }
    }
}

int main(int argc, char** argv) {
    // ... (argument parsing and preprocessing unchanged)

    // Load quadgrams - now using uppercase ('A')
    float* quadgrams = (float*)malloc(QUADGRAM_SIZE * sizeof(float));
    for (int i = 0; i < QUADGRAM_SIZE; ++i) quadgrams[i] = FLOOR_VALUE;

    FILE* fg = fopen("english_quadgrams.txt", "r");
    if (!fg) {
        fprintf(stderr, "Cannot open english_quadgrams.txt\n");
        return 1;
    }

    long long total_count = 0;
    char quad[5];
    long long count;
    while (fscanf(fg, "%4s %lld", quad, &count) == 2) {
        total_count += count;
    }
    rewind(fg);

    while (fscanf(fg, "%4s %lld", quad, &count) == 2) {
        int a = quad[0] - 'A';  // Uppercase
        int b = quad[1] - 'A';
        int c = quad[2] - 'A';
        int d = quad[3] - 'A';
        int idx = get_quadgram_index(a, b, c, d);
        quadgrams[idx] = logf((float)count / total_count);
    }
    fclose(fg);

    // ... (rest of main unchanged: copy to constant, launch kernel, decrypt, etc.)

    return 0;
}
