#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>



__global__ void lu_kernel(int n, double *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    //  moltiplicatori
    if (i == k && j > k && j < n)
        A[k*n + j] /= A[k*n + k];

    __syncthreads(); // sincronizzazione tra thread del blocco 

    // Aggiornamento sotto-matrice
    if (i > k && i < n && j > k && j < n)
        A[i*n + j] -= A[i*n + k] * A[k*n + j];
}


/* Inizializza la matrice */
void init_array(int n, double *A)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = i * n + j+1;
}


/* Funzione wrapper per chiamare il kernel */
void kernel_lu_cuda(int n, double *A)
{
    double *d_A;
    size_t size = n * n * sizeof(double);

    // Allocazione memoria device
    cudaMalloc((void **)&d_A, size);

    // Copia matrice su device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Definizione blocchi e griglie
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lancio kernel  (uno per ogni iterazione)
    for (int k = 0; k < n; k++) {
        lu_kernel<<<numBlocks, threadsPerBlock>>>(n, d_A, k);
        cudaDeviceSynchronize();  // sincronizzazione globale tra le iterazioni
    }

    // Copia risultato su host
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    // Free memoria device
    cudaFree(d_A);
}

/* Versione sequenziale per validazione */
void lu_seq(int n, double *A)
{
    int i, j, k;
    for (k = 0; k < n; k++) {
        // Divisione della riga k
        for (j = k + 1; j < n; j++)
            A[k*n + j] = A[k*n + j] / A[k*n + k];

        // Aggiornamento della matrice
        for (i = k + 1; i < n; i++)
            for (j = k + 1; j < n; j++)
                A[i*n + j] = A[i*n + j] - A[i*n + k] * A[k*n + j];
    }
}


/* Confronto matrici */
int confronta_matrici(int n, double *A, double *B, double tol)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (fabs(A[i * n + j] - B[i * n + j]) > tol)
                return 0;
    return 1;
}

int main()
{
    int n = 2048; // dimensione della matrice
    printf("Array dimension: %d\n", n);

    // Allocazione matrici
    double *A = (double *)malloc(n * n * sizeof(double));
    double *A_ref = (double *)malloc(n * n * sizeof(double));

    // Inizializzazione
    init_array(n, A);

    // Copia per versione sequenziale
    for (int i = 0; i < n * n; i++)
        A_ref[i] = A[i];

    
    // ================= GPU =================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_lu_cuda(n, A); //wrapper del kernel
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double gpu_time_s = milliseconds / 1000.0;  // converti in secondi
    printf("Tempo GPU: %f s\n", gpu_time_s);

    // ================= CPU =================
    clock_t cpu_start = clock();
    lu_seq(n, A_ref); // la versione sequenziale
    clock_t cpu_end = clock();
    double cpu_time_s = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("Tempo CPU: %f s\n", cpu_time_s);

    // Confronto
    double tol = 1e-6;
    int ok = confronta_matrici(n, A, A_ref, tol);

    if (ok)
        printf("VALIDAZIONE PASSATA\n");
    else
        printf("VALIDAZIONE NON PASSATA\n");

    // Free memoria
    free(A);
    free(A_ref);

    return 0;
}
