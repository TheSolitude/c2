// cu_hash_cracker.cu
// CUDA-based hash cracker: takes a hash and tests passwords using GPU
// Usage: ./cu_hash_cracker

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define HASH_LEN 64 // For SHA-256 hex string
#define PW_MAXLEN 8 // Max password length for brute-force
#define BLOCK_SIZE 256

__device__ void simple_sha256(const char* input, char* output) {
    // Placeholder: Replace with real SHA-256 implementation or use cuSHA
    // For demo, just copy input to output
    for (int i = 0; i < HASH_LEN; i++) output[i] = input[i % strlen(input)];
}

__device__ void idx_to_password(uint64_t idx, int pw_len, char* charset, int charset_len, char* out_pw) {
    for (int i = pw_len - 1; i >= 0; i--) {
        out_pw[i] = charset[idx % charset_len];
        idx /= charset_len;
    }
    out_pw[pw_len] = '\0';
}

__global__ void crack_kernel(const char* d_hash, int pw_len, char* charset, int charset_len, uint64_t total, int* d_found, uint64_t* d_found_idx, char* d_found_pw) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    char candidate[PW_MAXLEN+1];
    char hash[HASH_LEN+1];
    idx_to_password(idx, pw_len, charset, charset_len, candidate);
    simple_sha256(candidate, hash);
    if (strncmp(hash, d_hash, HASH_LEN) == 0) {
        *d_found = 1;
        *d_found_idx = idx;
        for (int i = 0; i < pw_len+1; i++) d_found_pw[i] = candidate[i];
    }
}

int main() {
    char target_hash[HASH_LEN+1];
    int pw_len;
    printf("Enter target hash (SHA-256 hex, 64 chars): ");
    fflush(stdout);
    fgets(target_hash, HASH_LEN+1, stdin);
    // Remove newline if present
    target_hash[strcspn(target_hash, "\n")] = 0;
    printf("Enter password length to brute-force (max %d): ", PW_MAXLEN);
    fflush(stdout);
    scanf("%d", &pw_len);
    if (pw_len < 1 || pw_len > PW_MAXLEN) {
        printf("Invalid password length.\n");
        return 1;
    }
    char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    int charset_len = strlen(charset);
    uint64_t total = 1;
    for (int i = 0; i < pw_len; i++) total *= charset_len;
    printf("Bruteforcing %llu combinations...\n", (unsigned long long)total);
    char* d_hash;
    char* d_charset;
    int* d_found;
    uint64_t* d_found_idx;
    char* d_found_pw;
    cudaMalloc(&d_hash, HASH_LEN);
    cudaMalloc(&d_charset, charset_len);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_found_idx, sizeof(uint64_t));
    cudaMalloc(&d_found_pw, PW_MAXLEN+1);
    cudaMemcpy(d_hash, target_hash, HASH_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_charset, charset, charset_len, cudaMemcpyHostToDevice);
    int found = 0;
    uint64_t found_idx = 0;
    cudaMemcpy(d_found, &found, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_idx, &found_idx, sizeof(uint64_t), cudaMemcpyHostToDevice);
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    crack_kernel<<<blocks, BLOCK_SIZE>>>(d_hash, pw_len, d_charset, charset_len, total, d_found, d_found_idx, d_found_pw);
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&found_idx, d_found_idx, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    char found_pw[PW_MAXLEN+1] = {0};
    cudaMemcpy(found_pw, d_found_pw, PW_MAXLEN+1, cudaMemcpyDeviceToHost);
    if (found) {
        printf("Password found: %s\n", found_pw);
    } else {
        printf("Password not found.\n");
    }
    cudaFree(d_hash); cudaFree(d_charset); cudaFree(d_found); cudaFree(d_found_idx); cudaFree(d_found_pw);
    return 0;
}
