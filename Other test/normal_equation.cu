#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 32

__global__ void matrixMultiply(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < m && col < k) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}


typedef float DataType;

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (!(row < numARows && col < numBColumns))
    return;

  // Iterate over row, and down column
  DataType tmp = 0;
  for (int i = 0; i < numAColumns; ++i) {
    // Accumulate results for a single element
    tmp += A[row * numAColumns + i] * B[i * numBColumns + col];
  }
  C[row * numBColumns + col] = tmp;
}

void inverseMatrixCPU(float *mat, int n) 
{
    int i, j, k;
    float temp;

    float *identity = (float*)malloc(n * n * sizeof(float));

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            if(i == j) {
                identity[i*n+j] = 1;
            } else {
                identity[i*n+j] = 0;
            }
        }
    }

    for(k = 0; k < n; k++) {
        temp = mat[k*n+k];
        for(j = 0; j < n; j++) {
            mat[k*n+j] /= temp;
            identity[k*n+j] /= temp;
        }
        for(i = 0; i < n   ; i++) {
            if(i == k) continue;
            temp = mat[i*n+k];
            for(j = 0; j < n; j++) {
                mat[i*n+j] -= mat[k*n+j]*temp;
                identity[i*n+j] -= identity[k*n+j]*temp;
            }
        }
    }
    
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            mat[i*n+j] = identity[i*n+j];
        }
    }
    
    free(identity);

}

void normalEquation(float *X, float *y, float *theta, int m, int n) {
    float *Xt, *XtX, *Xty;
    float *d_X, *d_y, *d_Xt, *d_XtX, *d_Xty;

    // allocate memory on host
    Xt = (float *)malloc(n * m * sizeof(float));
    XtX = (float *)malloc(n * n * sizeof(float));
    Xty = (float *)malloc(n * sizeof(float));

    // transpose X
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Xt[i * m + j] = X[j * n + i];
            
        }
    }



    // allocate memory on device
    cudaMalloc(&d_X, m * n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    cudaMalloc(&d_Xt, n * m * sizeof(float));
    cudaMalloc(&d_XtX, n * n * sizeof(float));
    cudaMalloc(&d_Xty, n * sizeof(float));

    // copy data to device
    cudaMemcpy(d_X, X, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xt, Xt, n * m * sizeof(float), cudaMemcpyHostToDevice);

    // compute XtX and Xty on device
    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    // dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);
    int BLOCKS_ROW = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int BLOCKS_COL = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 dimGrid(BLOCKS_COL, BLOCKS_ROW);

    gemm<<<dimGrid, dimBlock>>>(d_Xt, d_X, d_XtX, n, m, n);
    gemm<<<dimGrid, dimBlock>>>(d_Xt, d_y, d_Xty, n, m, 1);

    // copy data back to host
    cudaMemcpy(XtX, d_XtX, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Xty, d_Xty, n * sizeof(float), cudaMemcpyDeviceToHost);


    // calcola l'inversa di XtX
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i == j) {
                XtX[i * n + j] = 1.0 / XtX[i * n + j];
            } else {
                XtX[i * n + j] = 0.0;
            }
        }
    }

    // calcola theta usando la normal equation
    for(int i = 0; i < n; i++) {
        float sum = 0.0;
        for(int j = 0; j < n; j++) {
            sum += XtX[i * n + j] * Xty[j];
        }
        theta[i] = sum;
    }

    // free memory
    free(Xt);
    free(XtX);
    free(Xty);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_Xt);
    cudaFree(d_XtX);
    cudaFree(d_Xty);
}


int main() {

    float X[] = {1,2,3,4,5,6,7,8,9,10};
    float y[] = {3,5,7,9,11,13,15,17,19,21};
    float theta[2] = {0};

    normalEquation(X, y, theta, 10, 1);

    for (int i = 0; i < 2; i++) {
        printf("theta[%d] = %f\n", i, theta[i]);
    }

    return 0;

}