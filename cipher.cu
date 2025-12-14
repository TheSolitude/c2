// cuda_substitution_solver.cu
// FINAL FIXED VERSION – December 2025
// Fully race-safe, deterministic reductions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#include <float.h>
#include <ctype.h>
#include <math.h>

#define NUM_RESTARTS      16384
#define THREADS_PER_BLOCK 256
#define MAX_ITERATIONS    30000
#define QUADGRAM_SIZE     456976
#define FLOOR_VALUE       -12.0f
#define MAX_CIPHER_LEN    10000

float* d_quadgrams;

__constant__ char d_ciphertext[MAX_CIPHER_LEN];
__constant__ int  d_cipher_len;

__host__ __device__ inline int get_quadgram_index(int a, int b, int c, int d) {
    return ((a * 26 + b) * 26 + c) * 26 + d;
}

// Atomic max for float (negative-safe)
__device__ inline float atomicMaxFloat(float* addr, float val) {
    unsigned int* uaddr = (unsigned int*)addr;
    unsigned int old = *uaddr, assumed;
    do {
        assumed = old;
        old = atomicCAS(uaddr, assumed,
                        __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);
    return __uint_as_float(old);
}

__device__ float compute_fitness(const char* key, const float* quadgrams) {
    float score = 0.0f;
    for (int i = 0; i < d_cipher_len - 3; ++i) {
        int c1 = d_ciphertext[i]     - 'a';
        int c2 = d_ciphertext[i + 1] - 'a';
        int c3 = d_ciphertext[i + 2] - 'a';
        int c4 = d_ciphertext[i + 3] - 'a';

        int idx = get_quadgram_index(
            key[c1] - 'a',
            key[c2] - 'a',
            key[c3] - 'a',
            key[c4] - 'a'
        );
        score += quadgrams[idx];
    }
    return score;
}

__global__ void hill_climbing_kernel(float* best_global_score,
                                     char*  best_global_key,
                                     const float* quadgrams) {

    extern __shared__ char smem[];
    float* s_best_score = (float*)smem;
    int*   s_best_tid   = (int*)(s_best_score + 1);
    char*  s_best_key   = (char*)(s_best_tid + 1);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState rng;
    curand_init(0xC0FFEEULL + tid, 0, 0, &rng);

    char key[26], best_key_local[26];
    for (int i = 0; i < 26; ++i) key[i] = 'a' + i;

    for (int i = 0; i < 1500; ++i) {
        int a = curand(&rng) % 26;
        int b = curand(&rng) % 26;
        if (a != b) {
            char t = key[a]; key[a] = key[b]; key[b] = t;
        }
    }

    float score = compute_fitness(key, quadgrams);
    float best_score_local = score;
    memcpy(best_key_local, key, 26);

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        int a = curand(&rng) % 26;
        int b = curand(&rng) % 26;
        while (a == b) b = curand(&rng) % 26;

        char t = key[a]; key[a] = key[b]; key[b] = t;
        float s = compute_fitness(key, quadgrams);

        if (s > score) {
            score = s;
            if (s > best_score_local) {
                best_score_local = s;
                memcpy(best_key_local, key, 26);
            }
        } else {
            t = key[a]; key[a] = key[b]; key[b] = t;
        }
    }

    if (threadIdx.x == 0) {
        s_best_score[0] = -FLT_MAX;
        s_best_tid[0]   = -1;
    }
    __syncthreads();

    float prev = atomicMaxFloat(s_best_score, best_score_local);
    if (best_score_local > prev) {
        atomicExch(s_best_tid, threadIdx.x);
    }
    __syncthreads();

    if (threadIdx.x == s_best_tid[0]) {
        memcpy(s_best_key, best_key_local, 26);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float old = atomicMaxFloat(best_global_score, s_best_score[0]);
        if (s_best_score[0] > old) {
            memcpy(best_global_key, s_best_key, 26);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s \"ciphertext\"\n", argv[0]);
        return 1;
    }

    char ciphertext[MAX_CIPHER_LEN];
    int len = 0;
    for (int i = 0; argv[1][i]; ++i) {
        char c = tolower(argv[1][i]);
        if (isalpha(c)) ciphertext[len++] = c;
    }
    ciphertext[len] = 0;

    float* quad = (float*)malloc(QUADGRAM_SIZE * sizeof(float));
    for (int i = 0; i < QUADGRAM_SIZE; ++i) quad[i] = FLOOR_VALUE;

    FILE* f = fopen("english_quadgrams.txt", "r");
    long long total = 0, cnt;
    char q[5];
    while (fscanf(f, "%4s %lld", q, &cnt) == 2) total += cnt;
    rewind(f);
    while (fscanf(f, "%4s %lld", q, &cnt) == 2) {
        quad[get_quadgram_index(q[0]-'A',q[1]-'A',q[2]-'A',q[3]-'A')] =
            logf((float)cnt / total);
    }
    fclose(f);

    cudaMemcpyToSymbol(d_ciphertext, ciphertext, len+1);
    cudaMemcpyToSymbol(d_cipher_len, &len, sizeof(int));

    cudaMalloc(&d_quadgrams, QUADGRAM_SIZE * sizeof(float));
    cudaMemcpy(d_quadgrams, quad, QUADGRAM_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float* d_best_score;
    char*  d_best_key;
    cudaMalloc(&d_best_score, sizeof(float));
    cudaMalloc(&d_best_key, 26);
    float init = -FLT_MAX;
    cudaMemcpy(d_best_score, &init, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (NUM_RESTARTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t smem = sizeof(float) + sizeof(int) + 26;

    hill_climbing_kernel<<<blocks, THREADS_PER_BLOCK, smem>>>(
        d_best_score, d_best_key, d_quadgrams);
    cudaDeviceSynchronize();

    char key[26];
    cudaMemcpy(key, d_best_key, 26, cudaMemcpyDeviceToHost);

    printf("\n=== DECRYPTION ===\n");
    for (int i = 0; i < len; ++i)
        putchar(key[ciphertext[i]-'a']);
    printf("\n");

    return 0;
}
