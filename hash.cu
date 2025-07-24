#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#define HASH_LEN     32      // MD5 hex string length
#define PW_MAXLEN     8      // max password length
#define BLOCK_SIZE  256
// how many candidate passwords per batch (tunable)
#define BATCH_SIZE (1<<20)   // 1M passwords per host‑kernel launch

//---------------------------------------------------------------------------
// device MD5 (same as before)
//---------------------------------------------------------------------------

__device__ const uint32_t md5_k[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};
__device__ const uint32_t md5_r[64] = {
     7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
     5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
     4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
     6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

__device__ void to_hex(uint8_t *in, char *out) {
    const char *hex="0123456789abcdef";
    #pragma unroll
    for(int i=0;i<16;i++){
        out[2*i]   = hex[(in[i]>>4)&0xF];
        out[2*i+1] = hex[in[i]&0xF];
    }
    out[32]=0;
}

__device__ void md5(const char *msg, int len, char *out_hex) {
    uint8_t buf[64]={0};
    uint32_t a0=0x67452301, b0=0xefcdab89, c0=0x98badcfe, d0=0x10325476;
    // copy & pad
    #pragma unroll
    for(int i=0;i<len;i++) buf[i]=msg[i];
    buf[len]=0x80;
    uint64_t bits=len*8;
    buf[56]=bits; buf[57]=bits>>8; buf[58]=bits>>16; buf[59]=bits>>24;
    uint32_t *w=(uint32_t*)buf;
    uint32_t A=a0,B=b0,C=c0,D=d0;
    #pragma unroll
    for(int i=0;i<64;i++){
        uint32_t F,g;
        if(i<16){ F=(B&C)|(~B&D); g=i; }
        else if(i<32){ F=(D&B)|(~D&C); g=(5*i+1)%16; }
        else if(i<48){ F=B^C^D;       g=(3*i+5)%16; }
        else         { F=C^(B|~D);    g=(7*i)%16; }
        F += A + md5_k[i] + w[g];
        A=D; D=C; C=B;
        B += (F<<md5_r[i]) | (F>>(32-md5_r[i]));
    }
    uint8_t digest[16];
    ((uint32_t*)digest)[0]=a0+A;
    ((uint32_t*)digest)[1]=b0+B;
    ((uint32_t*)digest)[2]=c0+C;
    ((uint32_t*)digest)[3]=d0+D;
    to_hex(digest, out_hex);
}

__device__ void idx_to_pw(uint64_t idx,int pw_len,char *cs,int cs_len,char *out){
    for(int i=pw_len-1;i>=0;i--){
        out[i]=cs[idx%cs_len];
        idx/=cs_len;
    }
    out[pw_len]=0;
}

__device__ int my_strncmp(const char *a,const char *b,int n){
    for(int i=0;i<n;i++){
        if(a[i]!=b[i]) return a[i]-b[i];
    }
    return 0;
}

//---------------------------------------------------------------------------
// kernel with base offset
//---------------------------------------------------------------------------
__global__ void crack_kernel(const char *d_hash, int pw_len,
                             char *charset, int cs_len,
                             uint64_t total, int *d_found,
                             uint64_t *d_found_idx, char *d_found_pw,
                             uint64_t base)
{
    uint64_t tid = blockIdx.x*(uint64_t)blockDim.x + threadIdx.x;
    uint64_t idx = base + tid;
    if(idx>=total || *d_found) return;

    char pw[PW_MAXLEN+1], hash[HASH_LEN+1];
    idx_to_pw(idx, pw_len, charset, cs_len, pw);
    md5(pw, pw_len, hash);

    if(my_strncmp(hash, d_hash, HASH_LEN)==0){
        *d_found = 1;
        *d_found_idx = idx;
        #pragma unroll
        for(int i=0;i<=pw_len;i++) d_found_pw[i]=pw[i];
    }
}

//---------------------------------------------------------------------------
// host
//---------------------------------------------------------------------------
int main(){
    // 1) read inputs
    char target[HASH_LEN+2]; int pw_len;
    printf("MD5 hash (32 hex chars): ");
    fgets(target, sizeof(target), stdin);
    target[strcspn(target,"\n")]=0;

    printf("Password length (1-%d): ", PW_MAXLEN);
    scanf("%d",&pw_len);
    if(pw_len<1||pw_len>PW_MAXLEN) return puts("invalid len"),1;

    // 2) setup
    const char charset[]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    int cs_len = strlen(charset);
    uint64_t total=1;
    for(int i=0;i<pw_len;i++) total*=cs_len;

    // device allocations
    char *d_hash, *d_cs, *d_pw;
    int *d_found; uint64_t *d_fidx;
    cudaMalloc(&d_hash, HASH_LEN+1);
    cudaMalloc(&d_cs,  cs_len);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_fidx,  sizeof(uint64_t));
    cudaMalloc(&d_pw,    PW_MAXLEN+1);
    cudaMemcpy(d_hash, target, HASH_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cs, charset, cs_len, cudaMemcpyHostToDevice);
    int found=0; uint64_t fidx=0;
    cudaMemcpy(d_found,&found,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fidx,&fidx,sizeof(uint64_t),cudaMemcpyHostToDevice);

    // 3) loop batches
    uint64_t processed=0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for(uint64_t base=0; base<total && !found; base+=BATCH_SIZE){
        uint64_t remain = total-base;
        uint64_t batch = remain < BATCH_SIZE ? remain : BATCH_SIZE;
        int blocks = (batch + BLOCK_SIZE -1) / BLOCK_SIZE;

        crack_kernel<<<blocks, BLOCK_SIZE>>>(
            d_hash,pw_len,d_cs,cs_len,total,
            d_found,d_fidx,d_pw, base
        );
        cudaDeviceSynchronize();

        processed += batch;
        auto tn = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(tn - t0).count();
        double hps = processed / sec;
        // overwrite single line
        printf("\r%llu/%llu hashes processed — %.2f H/s",
               (unsigned long long)processed,
               (unsigned long long)total,
               hps);
        fflush(stdout);

        cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    }
    puts(""); // newline

    // 4) result
    if(found){
        char h_pw[PW_MAXLEN+1]={0};
        cudaMemcpy(h_pw, d_pw, PW_MAXLEN+1, cudaMemcpyDeviceToHost);
        printf("FOUND at index %llu: %s\n",
               (unsigned long long)fidx, h_pw);
    } else {
        puts("Not found.");
    }
    // cleanup
    cudaFree(d_hash); cudaFree(d_cs);
    cudaFree(d_found); cudaFree(d_fidx); cudaFree(d_pw);
    return 0;
}
