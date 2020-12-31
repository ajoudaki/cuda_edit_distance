#include <iostream>
#include <random>
#include <cassert>
#include <ctime>


typedef uint32_t dist_t;

typedef uint32_t result_t;

#define MIN(x,y) ((x) < (y) ? (x) : (y)) //calculate minimum between two values


// Convenience function for checking CUDA runtime API results
inline
cudaError_t checkCuda(cudaError_t result)
{
//#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
    return result;
}


__global__ void EditBlock(char *seq1, char *seq2, uint64_t chunk_size, uint64_t stride, result_t *result) {
    auto off1 = blockIdx.x * chunk_size + threadIdx.x * stride;
    auto off2 = blockIdx.y * chunk_size + threadIdx.y * stride;

    auto * dist = (dist_t*)malloc((chunk_size + 1) * sizeof(dist_t) );
    auto * dist_new = (dist_t*)malloc((chunk_size + 1) * sizeof(dist_t) );
    auto * tmp = dist;

    unsigned int  i,j,t, track;
    for(i=0;i<=chunk_size;i++) { // basis conditions
        dist[i] = i;
    }
    for (j=1;j<=chunk_size;j++) {
        // basis
        dist[0] = j - 1;
        dist_new[0] = j;
        for(i=1;i<=chunk_size;i++) {
            track = (seq1[off1+i-1] != seq2[off2 + j - 1]); // cost of mismatch = 1
            t = MIN(dist_new[i-1 ]+1, dist[i ] + 1);
            dist_new[i] = MIN(t, dist[i - 1] + track);
//          printf("%c,\t%c,\t%d,\t%d,\td(%d,%d)=%u\n", seq1[off1+i-1], seq2[off2+j-1], track, t, i,j, dist[i+pitch*j]);
        }
        tmp = dist;
        dist = dist_new;
        dist_new = tmp;
    }
    result += ((threadIdx.y+blockDim.y*blockIdx.y)* gridDim.x * blockDim.x + threadIdx.x+blockDim.x * blockIdx.x) * 3;
    result[0]= (result_t)off1;
    result[1] =(result_t) off2;
    result[2] = (result_t ) dist[chunk_size];
    free(dist);
    free(dist_new);
//    printf("%.32s,%.32s,%d\n", seq1 + off1, seq2 + off2, result[2]);
//    printf("result = %d, %d, %d\n", result[0], result[1], result[2]);
}


int main() {
    int64_t len = 1<<18;                        // seq len
    int64_t chunk_size = 1<<8;                 // consider chunk x chunk blocks of sequences
    int num_strides = 1;                    // number of strides per chunk
    int64_t stride = chunk_size / num_strides;
    unsigned int alphpabet_size = 4;

    dim3 strides2D = dim3(num_strides, num_strides );
    int64_t num_blocks = len / chunk_size;
    dim3 gridSize2D  = dim3(num_blocks, num_blocks );

    // load sequences into pinnable memory
    char *h1, *h2;
    auto bytes = (len+chunk_size) * sizeof(char); // zero-pad chnk_size bytes to avoid edge effects
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
    checkCuda(cudaMalloc(&s1, (len+chunk_size) * sizeof(char)) ) ;
    checkCuda( cudaMalloc(&s2, (len+chunk_size) * sizeof(char)) );
    checkCuda( cudaMemcpy(s1, h1, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(s2, h2, bytes, cudaMemcpyHostToDevice) );

    std::clock_t c_start = std::clock();
    // allocate results in unified memory
    result_t *res, *res_host;
    bytes = (num_blocks*num_strides)*(num_blocks*num_strides)*3*sizeof(result_t);
    checkCuda( cudaMallocManaged(&res, bytes) );
    std::cout << "Computing the edit distance blocks... \n" << std::flush;
    EditBlock<<< gridSize2D, strides2D >>>(s1, s2, chunk_size, stride, res);
    checkCuda( cudaDeviceSynchronize() );
    std::clock_t c_end = std::clock();

//    checkCuda( cudaMallocHost(&res_host, bytes));
//    checkCuda( cudaMemcpy(res_host, res, bytes, cudaMemcpyDeviceToHost));
//    checkCuda( cudaDeviceSynchronize() )
//    ;
    auto time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";
//    std::cout << "Writing output to file ... \n" << std::flush;
//    auto pFile = fopen ("myfile.txt","w");
//    assert(pFile && " file did not open for writing \n");
////    fprintf(pFile, "%s, %s, -1\n", h1, h2);
//    for (unsigned int i=0; i<bytes/sizeof(result_t); i += 3) {
//        fprintf(pFile, "%d, %d, %d\n", res[i+0], res[i+1], res[i+2]);
////        fprintf(pFile, "%lu, %lu, %lu\n", res_host[i+0], res_host[i+1], res_host[i+2]);
//    }
//    fclose(pFile);
}
