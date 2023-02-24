#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

#define THREADS_PER_BLOCK 1024

// Definizione della struttura dati per i dati del paziente
struct PatientData {
    float age;
    float sex;
    float smoker;
    float region;
    float charges;
};

// Funzione per calcolare la somma dei quadrati degli errori (SSE)
__global__ void sse_kernel(PatientData* data, float* theta, float* sse, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y = data[i].charges;
        float y_hat = theta[0] + theta[1] * data[i].age + theta[2] * data[i].sex + theta[3] * data[i].smoker + theta[4] * data[i].region;
        float error = y_hat - y;
        atomicAdd(sse, error * error);
    }
}

// Funzione per calcolare i parametri della regressione lineare usando il metodo dei minimi quadrati
void linear_regression(PatientData* data, int n, float* theta) {
    PatientData* d_data;
    float* d_theta;
    float* d_sse;

    // Allocazione della memoria sulla GPU
    cudaMalloc((void**)&d_data, n * sizeof(PatientData));
    cudaMalloc((void**)&d_theta, 5 * sizeof(float));
    cudaMalloc((void**)&d_sse, sizeof(float));

    // Copia dei dati e dei parametri sulla GPU
    cudaMemcpy(d_data, data, n * sizeof(PatientData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sse, 0, sizeof(float));

    // Calcolo del numero di blocchi e thread per blocco
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Esecuzione del kernel per il calcolo della SSE
    sse_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, d_theta, d_sse, n);

    // Copia del risultato dalla GPU alla CPU
    float sse;
    cudaMemcpy(&sse, d_sse, sizeof(float), cudaMemcpyDeviceToHost);

    // Calcolo dei parametri della regressione lineare
    int x_sum[5] = { n, 0, 0, 0, 0 };
    float y_sum = 0;
    float x_y_sum[5] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        x_sum[1] += data[i].age;
        x_sum[2] += data[i].sex;
        x_sum[3] += data[i].smoker;
        x_sum[4] += data[i].region;
        y_sum += data[i].charges;
        x_y_sum[    1] += data[i].age * data[i].charges;
        x_y_sum[2] += data[i].sex * data[i].charges;
        x_y_sum[3] += data[i].smoker * data[i].charges;
        x_y_sum[4] += data[i].region * data[i].charges;
    }

    float x_mean[5];
    x_mean[0] = 1;
    for (int i = 1; i < 5; i++) {
        x_mean[i] = x_sum[i] / n;
    }

    float y_mean = y_sum / n;

    float x_var[5] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        x_var[1] += pow(data[i].age - x_mean[1], 2);
        x_var[2] += pow(data[i].sex - x_mean[2], 2);
        x_var[3] += pow(data[i].smoker - x_mean[3], 2);
        x_var[4] += pow(data[i].region - x_mean[4], 2);
    }

    float cov_xy[5] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        cov_xy[1] += (data[i].age - x_mean[1]) * (data[i].charges - y_mean);
        cov_xy[2] += (data[i].sex - x_mean[2]) * (data[i].charges - y_mean);
        cov_xy[3] += (data[i].smoker - x_mean[3]) * (data[i].charges - y_mean);
        cov_xy[4] += (data[i].region - x_mean[4]) * (data[i].charges - y_mean);
    }

    float slope[5];
    slope[0] = y_mean - (cov_xy[1] / x_var[1]) * x_mean[1] - (cov_xy[2] / x_var[2]) * x_mean[2] - (cov_xy[3] / x_var[3]) * x_mean[3] - (cov_xy[4] / x_var[4]) * x_mean[4];
    slope[1] = cov_xy[1] / x_var[1];
    slope[2] = cov_xy[2] / x_var[2];
    slope[3] = cov_xy[3] / x_var[3];
    slope[4] = cov_xy[4] / x_var[4];

    // Copia dei parametri dalla GPU alla CPU
    cudaMemcpy(theta, slope, 5 * sizeof(float), cudaMemcpyHostToDevice);

    // Deallocazione della memoria sulla GPU
    cudaFree(d_data);
    cudaFree(d_theta);
    cudaFree(d_sse);

}

// Funzione per leggere i dati dal file CSV e salvarli in una matrice 2D
void read_csv(string filename, vector<vector<float>>& data) {
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<float> row;
            string field;
            stringstream ss(line);
            while (getline(ss, field, ',')) {
                row.push_back(stof(field));
            }
            data.push_back(row);
        }
            file.close();
    }
} 


int main() {
    // Lettura dei dati dal file CSV
    vector<vector<float>> csv_data;
    read_csv("data.csv", csv_data);
    // Conversione dei dati in un vettore di strutture
    int n = csv_data.size() - 1;
    PatientData* data = new PatientData[n];
    for (int i = 0; i < n; i++) {
        data[i].age = csv_data[i + 1][0];
        data[i].sex = csv_data[i + 1][1];
        data[i].smoker = csv_data[i + 1][2];
        data[i].region = csv_data[i + 1][3];
        data[i].charges = csv_data[i + 1][4];
    }

    // Calcolo della regressione lineare
    float* theta = new float[5];
    linear_regression(data, n, theta);

    // Stampa dei parametri della regressione lineare
    cout << "Intercept: " << theta[0] << endl;
    cout << "Slope for age: " << theta[1] << endl;
    cout << "Slope for sex: " << theta[2] << endl;
    cout << "Slope for smoker: " << theta[3] << endl;
    cout << "Slope for region: " << theta[4] << endl;

    // Calcolo del costo
    float sse = 0;
    for (int i = 0; i < n; i++) {
        float y_pred = theta[0] + theta[1] * data[i].age + theta[2] * data[i].sex + theta[3] * data[i].smoker + theta[4] * data[i].region;
        sse += pow(data[i].charges - y_pred, 2);
    }

    cout << "SSE: " << sse << endl;

    // Deallocazione della memoria
    delete[] data;
    delete[] theta;

    return 0;

} 