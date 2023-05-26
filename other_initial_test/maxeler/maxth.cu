#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // assumes device 0
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    printf("Maximum threads per block: %d\n", maxThreadsPerBlock);

    return 0;
}