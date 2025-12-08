#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <polybench.h>
#include "lu.h"

#define PAGEABLE

extern "C" void* polybench_alloc_data(long long, int);


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


static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  #ifdef DUMMY_MATRIX
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
	  if (i == j) A[i][j] = 40;
	  else A[i][j] = 1;
#else
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
#endif
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


/* Inizializza la matrice */
//void init_array(int n, double *A)
//{
//    for (int i = 0; i < n; i++)
//        for (int j = 0; j < n; j++)
//            A[i * n + j] = i * n + j+1;
//}

/* Stampa i primi 5 elementi della prima riga */
void print_first_5(double *A)
{
    for (int i = 0; i < 5; i++)
        printf("%f ", A[i]);
    printf("\n");
}

/* Funzione wrapper per chiamare il kernel */
void kernel_lu_cuda(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) //OLD: double* A
{
    double *d_A;
    size_t size = n * n * sizeof(double);

    // Allocazione memoria device
    cudaMalloc((void **)&d_A, size);

    // Copia matrice su device
    #if defined(PAGEABLE) || defined(PINNED)
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    #endif

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


#ifdef PAGEABLE
//     PAGEABLE VERSION
int main()
{
	int n = N;
    size_t size = n * n * sizeof(DATA_TYPE);
    //int n = 2048; // dimensione della matrice
    //printf("Array dimension: %d\n", n);
	POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

    // Allocazione matrici
    //double *A = (double *)malloc(n * n * sizeof(double));
    //double *A_lin = (double *)malloc(size); // per gpu
    double *A_ref = (double *)malloc(size); // per cpu

    // Inizializzazione
    init_array(n, POLYBENCH_ARRAY(A));

    DATA_TYPE *A_linear_ptr = (DATA_TYPE *)POLYBENCH_ARRAY(A);
    // Esegui la copia 1D CORRETTA
    for (int i = 0; i < n * n; i++){
        A_ref[i] = A_linear_ptr[i]; // Accesso 1D senza errori di tipo
        //A_lin[i] = A_linear_ptr[i];
    }

    // Prefetch della matrice alla GPU  (per performance)
    cudaMemPrefetchAsync(A, size, 0);
        
    // ================= GPU =================
    polybench_start_instruments;

    kernel_lu_cuda(n, POLYBENCH_ARRAY(A)); //wrapper del kernel
    polybench_stop_instruments;
    printf("Tempo GPU: ");
    polybench_print_instruments;
    //printf("Tempo GPU: %f s\n", gpu_time_s);

    // ================= CPU =================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    lu_seq(n, A_ref); // la versione sequenziale
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double cpu_time_s = milliseconds / 1000.0;
    printf("Tempo CPU: %f s\n", cpu_time_s);

    // Confronto
    double tol = 1e-6;
    int ok = confronta_matrici(n, A_linear_ptr, A_ref, tol);

    if (ok)
        printf("VALIDAZIONE PASSATA\n");
    else
        printf("VALIDAZIONE NON PASSATA\n");

    // Prefetch alla CPU (per la validazione)
    cudaMemPrefetchAsync(A, size, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    // Free memoria
    //free(A_lin);
    free(A_ref);
    POLYBENCH_FREE_ARRAY(A);
    return 0;
}
#endif


#ifndef N
	N = 4096
#endif


#ifdef UVM              //UVM version
   int main()
{
    //int n = 4096;
	int n = N; 
    size_t size = n * n * sizeof(double);

    //printf("Array dimension: %d\n", n);

    // ======= UVM ALLOCATION =======
    //double *A;
    double *A_ref;

	POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

    cudaMallocManaged(&A, size);
    //cudaMallocManaged(&A_ref, size);

    // Inizializzazione CPU
    //init_array(n, A);
	init_array(n, POLYBENCH_ARRAY(A));
	//print_array(n, POLYBENCH_ARRAY(A));

    //for (int i = 0; i < n*n; i++)
    //    A_ref[i] = A[i];

    // Prefetch della matrice alla GPU  (per performance)
    cudaMemPrefetchAsync(A, size, 0);

    // ========== GPU ==========
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_lu_cuda(n, POLYBENCH_ARRAY(A)); 
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
