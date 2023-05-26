

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

__global__ void matrixMultiplyWithBias(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < m && col < k) {
        for (int i = 0; i < n+1; i++) {
            if (i == n){
                sum += 1 * b[i*k+col];
            }
            else{
                sum += a[row * n+1 + i] * b[i * k + col];
            }
        }
        c[row * k + col] = sum;
    }
}


// Kernel per calcolare l'errore tra le predizioni e i valori effettivi
__global__ void compute_error(float* predictions, float* y, float* error, float size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<size){
        error[tid] = predictions[tid] - y[tid];
    }
}

__global__ void vectorSum(float* input,int size,float* output)
{
	int tid = threadIdx.x;

	int step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0 )
	{
		if (tid < number_of_threads || tid < size) // still alive?
		{
			const int fst = tid * step_size * 2;
			const int snd = fst + step_size;
			output[fst] += input[snd];
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}

__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, int rows, int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}