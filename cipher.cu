// cuda_substitution_solver.cu
// Latest fixed version resolving all compilation errors:
// - No direct atomicMax on float (not supported in CUDA)
// - Correct custom atomicMaxFloat using __float_as_uint / __uint_as_float (preserves bit pattern for proper ordering)
// - get_quadgram_index is __host__ __device__
// - Quadgrams in global memory
// - Robust block + global reduction

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#include <float.h>
#include <ctype.h>
#include <math.h>

#define NUM_RESTARTS      16384
#define THREADS_PER_BLOCK 256
#define MAX_ITERATIONS    20000
#define QUADGRAM_SIZE     456976
#define FLOOR_VALUE       -12.0f
#define MAX_CIPHER_LEN    10000

float* d_quadgrams;

__constant__ char d_ciphertext[MAX_CIPHER_LEN];
__constant__ int  d_cipher_len;

__host__ __device__ inline int get_quadgram_index(int a, int b, int c, int d) {
    return ((a * 26 + b) * 26 + c) * 26 + d;
}

// Correct atomicMax for float (using unsigned int bit representation)
__device__ inline float atomicMaxFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;

    do {
        assumed = old;
        float current = __uint_as_float(assumed);
        if (current >= val) break;  // No need to update
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);

    return __uint_as_float(old);
}

__device__ float compute_fitness(const char* key, const float* quadgrams) {
    float score = 0.0f;
    for (int i = 0; i < d_cipher_len - 3; ++i) {
        int c1 = d_ciphertext[i] - 'a';
        int c2 = d_ciphertext[i + 1] - 'a';
        int c3 = d_ciphertext[i + 2] - 'a';
        int c4 = d_ciphertext[i + 3] - 'a';

        char p1 = key[c1];
        char p2 = key[c2];
        char p3 = key[c3];
        char p4 = key[c4];

        int idx = get_quadgram_index(p1 - 'a', p2 - 'a', p3 - 'a', p4 - 'a');
        score += quadgrams[idx];
    }
    return score;
}

__global__ void hill_climbing_kernel(float* best_global_score,
                                     char* best_global_key,
                                     const float* quadgrams) {
    extern __shared__ char shared_mem[];

    float* s_best_score = (float*)shared_mem;
    char*  s_best_key   = (char*)(s_best_score + 1);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(tid + (unsigned long long)clock64(), 0, 0, &state);

    char key[26];
    char local_best_key[26];
    for (int i = 0; i < 26; ++i) key[i] = 'a' + i;

    for (int i = 0; i < 1000; ++i) {
        int a = curand(&state) % 26;
        int b = curand(&state) % 26;
        if (a != b) {
            char tmp = key[a];
            key[a] = key[b];
            key[b] = tmp;
        }
    }

    float current_score = compute_fitness(key, quadgrams);
    float local_best_score = current_score;
    memcpy(local_best_key, key, 26);

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        int a = curand(&state) % 26;
        int b = curand(&state) % 26;
        while (a == b) b = curand(&state) % 26;

        char tmp = key[a];
        key[a] = key[b];
        key[b] = tmp;

        float new_score = compute_fitness(key, quadgrams);

        if (new_score > current_score) {
            current_score = new_score;
            if (new_score > local_best_score) {
                local_best_score = new_score;
                memcpy(local_best_key, key, 26);
            }
        } else {
            tmp = key[a];
            key[a] = key[b];
            key[b] = tmp;
        }
    }

    // Block-level reduction
    if (threadIdx.x == 0) {
        s_best_score[0] = -FLT_MAX;
    }
    __syncthreads();

    atomicMaxFloat(s_best_score, local_best_score);

    __syncthreads();

    if (local_best_score == s_best_score[0]) {
        memcpy(s_best_key, local_best_key, 26);
    }

    __syncthreads();

    // Global update (one thread per block)
    if (threadIdx.x == 0) {
        float old_global = atomicMaxFloat(best_global_score, s_best_score[0]);
        if (s_best_score[0] > old_global) {
            memcpy(best_global_key, s_best_key, 26);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s \"ciphertext in quotes\"\n", argv[0]);
        return 1;
    }

    const char* raw_ciphertext = argv[1];

    char ciphertext[MAX_CIPHER_LEN];
    int len = 0;
    for (int i = 0; raw_ciphertext[i] && len < MAX_CIPHER_LEN - 1; ++i) {
        if (isalpha(raw_ciphertext[i])) {
            ciphertext[len++] = tolower(raw_ciphertext[i]);
        }
    }
    ciphertext[len] = '\0';

    printf("Processed ciphertext length: %d characters\n", len);

    float* quadgrams = (float*)malloc(QUADGRAM_SIZE * sizeof(float));
    for (int i = 0; i < QUADGRAM_SIZE; ++i) quadgrams[i] = FLOOR_VALUE;

    FILE* fg = fopen("english_quadgrams.txt", "r");
    if (!fg) {
        fprintf(stderr, "Error: Cannot open 'english_quadgrams.txt'\n");
        free(quadgrams);
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
        int a = quad[0] - 'A';
        int b = quad[1] - 'A';
        int c = quad[2] - 'A';
        int d = quad[3] - 'A';
        int idx = get_quadgram_index(a, b, c, d);
        quadgrams[idx] = logf((float)count / total_count);
    }
    fclose(fg);
    printf("Loaded quadgram statistics\n");

    cudaMemcpyToSymbol(d_ciphertext, ciphertext, len + 1);
    cudaMemcpyToSymbol(d_cipher_len, &len, sizeof(int));

    cudaMalloc(&d_quadgrams, QUADGRAM_SIZE * sizeof(float));
    cudaMemcpy(d_quadgrams, quadgrams, QUADGRAM_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(quadgrams);

    float* d_best_score;
    char*  d_best_key;
    cudaMalloc(&d_best_score, sizeof(float));
    cudaMalloc(&d_best_key, 26 * sizeof(char));

    float init_score = -FLT_MAX;
    cudaMemcpy(d_best_score, &init_score, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (NUM_RESTARTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem_size = sizeof(float) + 26 * sizeof(char);

    printf("Launching %d restarts (%d blocks × %d threads)\n", NUM_RESTARTS, blocks, THREADS_PER_BLOCK);

    hill_climbing_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        d_best_score, d_best_key, d_quadgrams);

    cudaDeviceSynchronize();

    float best_score;
    char best_key[26];
    cudaMemcpy(&best_score, d_best_score, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_key, d_best_key, 26, cudaMemcpyDeviceToHost);

    char plaintext[MAX_CIPHER_LEN];
    for (int i = 0; i < len; ++i) {
        plaintext[i] = best_key[ciphertext[i] - 'a'];
    }
    plaintext[len] = '\0';

    printf("\n=== BEST RESULT ===\n");
    printf("Score: %.2f\n\n", best_score);
    printf("Decrypted text:\n%s\n\n", plaintext);

    printf("Key (cipher → plain):\n");
    for (int i = 0; i < 26; ++i) {
        printf("%c → %c  ", 'a' + i, best_key[i]);
        if (i % 8 == 7) printf("\n");
    }
    printf("\n");

    cudaFree(d_quadgrams);
    cudaFree(d_best_score);
    cudaFree(d_best_key);

    return 0;
}
