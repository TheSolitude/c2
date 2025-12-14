// cuda_substitution_solver.cu
// CUDA implementation of parallel random-restart hill climbing for monoalphabetic substitution cipher
// Using quadgram statistics for fitness scoring
// Compile: nvcc -O3 cuda_substitution_solver.cu -o cuda_substitution_solver -lcurand
// Run: ./cuda_substitution_solver "your ciphertext here"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#include <float.h>
#include <ctype.h>

#define NUM_RESTARTS 16384      // Number of parallel restarts (threads) - adjust based on your GPU
#define THREADS_PER_BLOCK 256
#define MAX_ITERATIONS 20000    // Hill climbing steps per restart
#define QUADGRAM_SIZE 456976    // 26^4
#define FLOOR_VALUE -12.0f      // Floor for unseen quadgrams

__constant__ float d_quadgrams[QUADGRAM_SIZE];
__constant__ char d_ciphertext[10000];  // Max ciphertext length - adjust if needed
__constant__ int d_cipher_len;

__device__ int get_quadgram_index(int a, int b, int c, int d) {
    return ((a * 26 + b) * 26 + c) * 26 + d;
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

__global__ void hill_climbing_kernel(float* best_global_score, char* best_global_key, int* best_thread_id) {
    __shared__ float s_best_score;
    __shared__ int s_best_thread;
    __shared__ char s_best_key[26];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(tid + clock64(), 0, 0, &state);

    // Key: mapping from ciphertext letter index (0-25) to plaintext char ('a' + x)
    char key[26];
    char best_key[26];
    for (int i = 0; i < 26; ++i) key[i] = 'a' + i;

    // Randomize initial key with many swaps
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

        // Swap
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
            // Revert swap
            temp = key[a];
            key[a] = key[b];
            key[b] = temp;
        }
    }

    // Block-level reduction to find best in block
    if (threadIdx.x == 0) {
        s_best_score = -FLT_MAX;
    }
    __syncthreads();

    atomicMax(&s_best_score, local_best_score);  // Note: atomicMax for float needs care, but works on modern CUDA

    __syncthreads();

    if (local_best_score == s_best_score) {
        atomicMax(best_global_score, local_best_score);
        if (local_best_score == *best_global_score) {
            *best_thread_id = tid;
            memcpy(best_global_key, best_key, 26);
        }
    }
}

int main(int argc, char** argv) {
    const char* raw_ciphertext;
    if (argc >= 2) {
        raw_ciphertext = argv[1];
    } else {
        printf("Usage: %s \"ciphertext\"\n", argv[0]);
        return 1;
    }

    // Preprocess ciphertext: lowercase, keep only letters
    char ciphertext[10000];
    int len = 0;
    for (int i = 0; raw_ciphertext[i]; ++i) {
        if (isalpha(raw_ciphertext[i])) {
            ciphertext[len++] = tolower(raw_ciphertext[i]);
        }
    }
    ciphertext[len] = '\0';
    printf("Processed ciphertext length: %d\n", len);

    if (len < 50) {
        printf("Warning: Very short ciphertext - success not guaranteed.\n");
    }

    // Load quadgram log probabilities
    float* quadgrams = (float*)malloc(QUADGRAM_SIZE * sizeof(float));
    for (int i = 0; i < QUADGRAM_SIZE; ++i) quadgrams[i] = FLOOR_VALUE;

    FILE* fg = fopen("english_quadgrams.txt", "r");
    if (!fg) {
        fprintf(stderr, "Cannot open english_quadgrams.txt - download from http://practicalcryptography.com/media/cryptanalysis/files/quadgrams.txt\n");
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
    printf("Loaded quadgram statistics (total count: %lld)\n", total_count);

    // Copy to GPU constant memory
    cudaMemcpyToSymbol(d_quadgrams, quadgrams, QUADGRAM_SIZE * sizeof(float));
    cudaMemcpyToSymbol(d_ciphertext, ciphertext, len + 1);
    cudaMemcpyToSymbol(d_cipher_len, &len, sizeof(int));

    free(quadgrams);

    // Allocate global best
    float* d_best_score;
    char* d_best_key;
    int* d_best_thread;
    cudaMalloc(&d_best_score, sizeof(float));
    cudaMalloc(&d_best_key, 26 * sizeof(char));
    cudaMalloc(&d_best_thread, sizeof(int));

    float h_best_score = -FLT_MAX;
    cudaMemcpy(d_best_score, &h_best_score, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (NUM_RESTARTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d restarts (%d blocks x %d threads)\n", NUM_RESTARTS, blocks, THREADS_PER_BLOCK);

    hill_climbing_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_best_score, d_best_key, d_best_thread);
    cudaDeviceSynchronize();

    // Retrieve best
    char best_key[26];
    cudaMemcpy(&h_best_score, d_best_score, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_key, d_best_key, 26, cudaMemcpyDeviceToHost);

    // Decrypt
    char plaintext[10000];
    for (int i = 0; i < len; ++i) {
        int c = ciphertext[i] - 'a';
        plaintext[i] = best_key[c];
    }
    plaintext[len] = '\0';

    printf("\nBest score: %.2f\n", h_best_score);
    printf("Decrypted text:\n%s\n\n", plaintext);

    printf("Key (cipher -> plain):\n");
    for (int i = 0; i < 26; ++i) {
        if (i % 8 == 0 && i > 0) printf("\n");
        printf("%c->%c ", 'a' + i, best_key[i]);
    }
    printf("\n");

    cudaFree(d_best_score);
    cudaFree(d_best_key);
    cudaFree(d_best_thread);

    return 0;
}
