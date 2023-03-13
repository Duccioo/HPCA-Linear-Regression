#include <iostream>
#include <cuda_runtime.h>

// Numero di punti e coefficienti
const int N = 10000000;
const int num_coef = 2;

// Struttura dei dati di input
struct Data
{
    float x[num_coef];
    float y;
};

// Funzione per generare i dati di input casuali
void generate_data(Data* data, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < num_coef; j++)
        {
            data[i].x[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        data[i].y = static_cast<float>(rand()) / RAND_MAX;
    }
   
}

// generate input data for regression
void generate_data2(Data* data, int n){ 
	printf("Input data is a linear function: 0.5 *x - 27 with randon noise added.\n");
	int i;
	for(i = 0; i<n; i++){
		for (int j = 0; j < num_coef; j++)
        {
            data[i].x[j] = static_cast<float>(rand()) / RAND_MAX;
        }
		data[i].y = 0.5  * (float)i - 27 + (float)rand()/RAND_MAX - 0.5; // linear function with some random variations
	}
}

// Funzione per calcolare il prodotto scalare tra due vettori
__device__ float dot(float* a, float* b, int n)
{
    float result = 0;
    for (int i = 0; i < n; i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

// Funzione kernel per il calcolo dei coefficienti della regressione lineare
__global__ void linear_regression_kernel(Data* data, float* coef)
{
    // Indice del thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Matrice di input
    float A[num_coef][num_coef];
    for (int j = 0; j < num_coef; j++)
    {
        for (int k = 0; k < num_coef; k++)
        {
            A[j][k] = dot(data[i].x, data[k].x, num_coef);
        }
    }

    // Vettore di output
    float b[num_coef];
    for (int j = 0; j < num_coef; j++)
    {
        b[j] = dot(data[i].x, &data[i].y, num_coef);
    }

    // Risoluzione del sistema lineare
    for (int j = 0; j < num_coef; j++)
    {
        for (int k = j + 1; k < num_coef; k++)
        {
            float factor = A[k][j] / A[j][j];
            for (int l = j; l < num_coef; l++)
            {
                A[k][l] -= factor * A[j][l];
            }
            b[k] -= factor * b[j];
        }
    }
    for (int j = num_coef - 1; j >= 0; j--)
    {
        for (int k = j - 1; k >= 0; k--)
        {
            float factor = A[k][j] / A[j][j];
            for (int l = j; l >= 0; l--)
            {
                A[k][l] -= factor * A[j][l];
            }
            b[k] -= factor * b[j];
        }
        coef[j] = b[j] / A[j][j];
    }
}

// Funzione per la regressione lineare su CPU
void linear_regression_cpu(Data* data, float* coef)
{
    float A[num_coef][num_coef];
    float b[num_coef];

    // Calcolo la matrice di input e il vettore di output
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < num_coef; j++)
        {
            for (int k = 0; k < num_coef; k++)
            {
                A[j][k] += data[i].x[j] * data[i].x[k];
            }
            b[j] += data[i].x[j] * data[i].y;
        }
    }
    
    // Risoluzione del sistema lineare
    for (int j = 0; j < num_coef; j++)
    {
        for (int k = j + 1; k < num_coef; k++)
        {
            float factor = A[k][j] / A[j][j];
            for (int l = j; l < num_coef; l++)
            {
                A[k][l] -= factor * A[j][l];
            }
            b[k] -= factor * b[j];
        }
    }
    for (int j = num_coef - 1; j >= 0; j--)
    {
        for (int k = j - 1; k >= 0; k--)
        {
            float factor = A[k][j] / A[j][j];
            for (int l = j; l >= 0; l--)
            {
                A[k][l] -= factor * A[j][l];
            }
            b[k] -= factor * b[j];
        }
        coef[j] = b[j] / A[j][j];
        printf("coef %f\n", coef[j]);
    }

    
}

int main()
{
    // Allocazione della memoria per i dati di input e i coefficienti
    Data* data = new Data[N];
    float* coef = new float[num_coef];
    // Generazione dei dati di input casuali
    generate_data2(data, N);

    // Misurazione del tempo di esecuzione su CPU
    clock_t start_cpu = clock();
    linear_regression_cpu(data, coef);
    clock_t end_cpu = clock();
    std::cout << "Tempo di esecuzione su CPU: " << static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC << " secondi" << std::endl;

    // Allocazione della memoria per i dati di input e i coefficienti sulla GPU
    Data* data_gpu;
    cudaMalloc(&data_gpu, N * sizeof(Data));
    float* coef_gpu;
    cudaMalloc(&coef_gpu, num_coef * sizeof(float));

    // Copia dei dati di input dalla CPU alla GPU
    cudaMemcpy(data_gpu, data, N * sizeof(Data), cudaMemcpyHostToDevice);

    // Configurazione della griglia dei thread e dei blocchi
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Misurazione del tempo di esecuzione su GPU
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);
    linear_regression_kernel<<<numBlocks, threadsPerBlock>>>(data_gpu, coef_gpu);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, end_gpu);
    std::cout << "Tempo di esecuzione su GPU: " << gpu_time_ms / 1000.0 << " secondi" << std::endl;

    // Copia dei coefficienti dalla GPU alla CPU
    cudaMemcpy(coef, coef_gpu, num_coef * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocazione della memoria
    delete[] data;
    delete[] coef;
    cudaFree(data_gpu);
    cudaFree(coef_gpu);
}
