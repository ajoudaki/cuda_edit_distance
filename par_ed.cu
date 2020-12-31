#include <iostream>
#include <random>
#include <cassert>
#include <ctime>

//#define DEBUG

typedef float dist_t;
typedef dist_t result_t;

// Convenience function for checking CUDA runtime API results
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

#define MIN(x,y) ((x) < (y) ? (x) : (y)) //calculate minimum between two values


__global__ void upper_diagonal(const char *seq1, const char *seq2, unsigned int l, result_t *d0, result_t* d1, result_t* d2) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<=0 or i>=l) {
        d2[i] = l;
    } else {
        result_t t = seq1[i-1]!=seq2[l-i-1];
        d2[i] = MIN(t + d0[i-1], MIN(d1[i],d1[i-1])+1 );
    }
}

__global__ void lower_diagonal(const char *seq1, const char *seq2, unsigned int len, unsigned int l, result_t *d0, result_t* d1, result_t* d2) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<=0 or i>=l) {
        d2[i] = l;
    } else {
        result_t t = seq1[i-1]!=seq2[l-i-1];
        d2[i] = MIN(t + d0[i-1], MIN(d1[i],d1[i-1])+1 );
    }
}

void levenshtein_distance(char *s1, char *s2, unsigned int l1, unsigned int l2, std::vector<std::vector<unsigned int>> &dist) {
    dist = std::vector<std::vector<unsigned int>>(l2+1, std::vector<unsigned int>(l1+1, (l1+l2)*2));
    for(int i=0;i<=l1;i++) {
        dist[0][i] = i;
    }
    for(int j=0;j<=l2;j++) {
        dist[j][0] = j;
    }
    for (int j=1;j<=l1;j++) {
        for(int i=1;i<=l2;i++) {
            int track, t;
            if(s1[i-1] == s2[j-1]) {
                track= 0;
            } else {
                track = 1;
            }
            t = MIN((dist[i-1][j]+1),(dist[i][j-1]+1));
            dist[i][j] = MIN(t,(dist[i-1][j-1]+track));
        }
    }
#if defined(DEBUG) || defined(_DEBUG)
        std::cout<<"The Levinstein distance is:"<<dist[l2][l1] << std::endl;
#endif
}

int main() {
    uint64_t blocksize = 128;
    uint64_t len = (1<<14);
    unsigned int alphpabet_size = 4;

    // load sequences into pinnable memory
    char *h1, *h2;
    auto bytes = (len+1) * sizeof(char); // zero-pad by 1 byte to add end-of-sequence-char:0
    checkCuda(cudaMallocHost(&h1, bytes));
    checkCuda(cudaMallocHost(&h2, bytes ));
    for (int i=0; i < len; i++) {
        int r = rand() ;
        h1[i] = (char)(r % alphpabet_size + (int)'A');
        r = rand();
        h2[i]= (char)(r % alphpabet_size + (int)'A') ;
    }
    h1[len] = h2[len] = 0;

    // move to device memory
    char *s1, *s2;
    checkCuda(cudaMalloc(&s1, bytes) ) ;
    checkCuda( cudaMalloc(&s2, bytes ) );
    checkCuda( cudaMemcpy(s1, h1, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(s2, h2, bytes, cudaMemcpyHostToDevice) );

    std::clock_t c_start, c_end;


    result_t *d0, *d1, *d2, *tmp;
    uint64_t pitch = len + blocksize;                       // pad distances by at least blocksize
    bytes = 3* pitch * sizeof(result_t) ;                   // allocate 3 x pitch * size type
    checkCuda( cudaMalloc(&d0, bytes) );       // allocate on unified memory
    checkCuda(cudaMemset(d0, 2* len, bytes));   // initialize distances to maximum value 2*len
    d1 = d0 + pitch;
    d2 = d1 + pitch;

    c_start = std::clock();
    printf("Computing the edit distance blocks... \n" );


    uint64_t l;
    for (l=0; l<=len; l++) {
        upper_diagonal<<< (l + blocksize) / blocksize, blocksize >>>(s1, s2, l, d0, d1, d2);
        checkCuda( cudaDeviceSynchronize() );
        tmp = d0; d0 = d1; d1 = d2; d2 = tmp;
    }
    dist_t *d_host;
    bytes = (len+1)* sizeof (dist_t);
    cudaMallocHost(&d_host, bytes);
    checkCuda( cudaMemcpy(d_host, d1, bytes, cudaMemcpyHostToDevice) );

    auto gpu_time = 1.0 * (std::clock()-c_start) / CLOCKS_PER_SEC;
    printf("GPU time: %f s\n", gpu_time);

#if defined(DEBUG) || defined(_DEBUG)
        // check results
        std::vector<std::vector<unsigned int>> dist;
        c_start = std::clock();

        levenshtein_distance(h1, h2, len, len, dist);

        auto cpu_time = 1.0 * (std::clock()-c_start) / CLOCKS_PER_SEC;
        printf("CPU time: %f s\n", cpu_time);

        printf("speed up %f\n", cpu_time/gpu_time);
        int mismatch = 0;
        for (int i=0; i<=len; i++) {
            if (d_host[i] != dist[i][len-i]) {
                mismatch++;
            }
        }
        printf("number of mismatches = %d, out of %d\n",mismatch, len);
        assert(mismatch==0 && " mismatch found ");
#endif

#if defined(_DEBUG)
        printf("final dist:\n");
        for (int i=0; i<=len; i++) {
            printf("%d ",d_host[i]);
        }
        printf("\nED diag %lu =\n",len);
        for (int i=0; i<=len; i++) {
            printf("%d,",dist[i][len-i]);
        }
        printf("\n");
        printf("%.32s...,%.32s...\n", h1, h2);

#endif

}