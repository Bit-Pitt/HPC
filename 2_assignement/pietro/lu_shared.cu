#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

#ifndef N
#define N 2048
#endif

//Questo non funziona
__global__ void lu_kernel_shared(int n, double *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x +(k+1);   
    int i = blockIdx.y * blockDim.y + threadIdx.y +(k+1);

    if (i >= n || j >= n)
        return; 

    __shared__ double pivot_row[BLOCK_SIZE];  
    __shared__ double pivot_col[BLOCK_SIZE];  

  
    if (threadIdx.y == 0  && j < n) 
    {
       pivot_row[threadIdx.x] = A[k*n + j];              
    }
  
    if (threadIdx.x == 0 && i < n)
    {
        pivot_col[threadIdx.y] = A[i*n + k];        
    }

    __syncthreads();

   
    double Aik = pivot_col[threadIdx.y];  
    double Akj = pivot_row[threadIdx.x];  
    A[i*n + j] -= Aik * Akj;
    
}

//Kernel 1D
__global__ void kernel_mul(int n, double* A, int k)
{
    __shared__ double pivot;
    int i = blockIdx.x * blockDim.x + threadIdx.x + (k+1);   

    if (threadIdx.x==0)     
        pivot = A[k*n+k];

    if (i < n)  
    {   
        A[i*n + k] /= pivot;
    }
}

// per la sottomatrice
__global__ void kernel_lu(int n, double *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k+1);      
    int i = blockIdx.y * blockDim.y + threadIdx.y + (k+1);

    if (i < n && j < n)     
    {
        A[i*n + j] -= A[i*n + k] * A[k*n + j];
    }
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
    #if defined(PAGEABLE) || defined(PINNED)
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    #endif

    dim3 threads1D(BLOCK_SIZE);
    dim3 threads2D(BLOCK_SIZE, BLOCK_SIZE);

    for (int k = 0; k < n; k++)
    {
        int remaining_rows = n - (k + 1);
        int blocks1D = (remaining_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;     
        kernel_mul<<<blocks1D, threads1D>>>(n, d_A, k);
        cudaDeviceSynchronize();

        dim3 blocks2D(
            (remaining_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (remaining_rows + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        lu_kernel_shared<<<blocks2D, threads2D>>>(n, d_A, k);
        cudaDeviceSynchronize();
    }

    

    // Copia risultato su host
    #if defined(PAGEABLE) || defined(PINNED)
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    #endif

    // Free memoria device
    cudaFree(d_A);
}

/* Versione sequenziale per validazione */
void lu_seq(int n, double *A)
{
    int i, j, k;
    for (k = 0; k < n; k++) {

        //moltiplicatori
        for (i = k+1; i < n; i++)
            A[i*n+k] = A[i*n+k] / A[k*n+k];   

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


#ifdef PAGEABLE
//     PAGEABLE VERSION
int main()
{
    int n = N; // dimensione della matrice
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
#endif



#ifdef PINNED
int main()
{
    int n = N; // dimensione della matrice 
    printf("Array dimension: %d\n", n);

    // -----------------------------
    // Allocazione matrici PINNED
    // -----------------------------
    double *A, *A_ref;
    cudaMallocHost((void **)&A, n * n * sizeof(double));
    cudaMallocHost((void **)&A_ref, n * n * sizeof(double));

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


    // -----------------------------
    // Free memoria PINNED
    // -----------------------------
    cudaFreeHost(A);
    cudaFreeHost(A_ref);

    return 0;
}
#endif





#ifdef UVM              //UVM version
   int main()
{
    int n = N;
    size_t size = n * n * sizeof(double);

    printf("Array dimension: %d\n", n);

    // ======= UVM ALLOCATION =======
    double *A;
    double *A_ref;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&A_ref, size);

    // Inizializzazione CPU
    init_array(n, A);
    for (int i = 0; i < n*n; i++)
        A_ref[i] = A[i];

    // Prefetch della matrice alla GPU  (per performance)
    cudaMemPrefetchAsync(A, size, 0);

    // ========== GPU ==========
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_lu_cuda(n, A); 
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double gpu_time_s = milliseconds / 1000.0;
    printf("Tempo GPU: %f s\n", gpu_time_s);

    // Prefetch alla CPU (per la validazione)
    cudaMemPrefetchAsync(A, size, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    // Liberazione UVM
    cudaFree(A);
    cudaFree(A_ref);

    return 0;
}

#endif