#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

#include <polybench.h>
#include "lu.h"

extern "C" void* polybench_alloc_data(long long, int);


//Kernel per i moltiplicatori
__global__ void kernel_mul(int n, double* A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    double pivot = A[k*n + k];
    // moltiplicatori
    if (j == k && i > k && i < n)
        A[i*n + k] /= pivot;
}

__global__ void lu_kernel_base(int n, double *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Aggiornamento sotto-matrice
    if (i > k && i < n && j > k && j < n)
        A[i*n + j] -= A[i*n + k] * A[k*n + j];
}


/* Inizializza la matrice */
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

    // Definizione blocchi e griglie
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (n + BLOCK_SIZE- 1) / BLOCK_SIZE);

    // Lancio kernel  (2 per ogni iterazione)
    for (int k = 0; k < n; k++)
    {
        kernel_mul<<<numBlocks, threadsPerBlock>>>(n, d_A, k);      //kernel per i moltiplicatori
        cudaDeviceSynchronize();      

        lu_kernel_base<<<numBlocks, threadsPerBlock>>>(n, d_A, k);  //per aggiornamento sottomatrice
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
    size_t size = n * n * sizeof(DATA_TYPE);
    printf("Array dimension: %d\n", n);

    // Allocazione matrici
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    double *A_ref = (double *)malloc(size);

    // Inizializzazione
    init_array(n, POLYBENCH_ARRAY(A));

    DATA_TYPE *A_linear_ptr = (DATA_TYPE *)POLYBENCH_ARRAY(A);
    // Esegui la copia 1D per versione sequenziale
    for (int i = 0; i < n * n; i++){
        A_ref[i] = A_linear_ptr[i];
    }

    
    // ================= GPU =================
    polybench_start_instruments;

    kernel_lu_cuda(n, (double*)POLYBENCH_ARRAY(A)); //wrapper del kernel
    polybench_stop_instruments;
    printf("Tempo GPU: ");
    polybench_print_instruments;


    // ================= CPU =================
    clock_t cpu_start = clock();
    lu_seq(n, A_ref); // la versione sequenziale
    clock_t cpu_end = clock();
    double cpu_time_s = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("Tempo CPU: %f s\n", cpu_time_s);

    // Confronto
    double tol = 1e-6;
    int ok = confronta_matrici(n, A_linear_ptr, A_ref, tol);

    if (ok)
        printf("VALIDAZIONE PASSATA\n");
    else
        printf("VALIDAZIONE NON PASSATA\n");

    // Free memoria
    POLYBENCH_FREE_ARRAY(A);
    free(A_ref);

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
	POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

    cudaMallocManaged(&A, size);

    // Inizializzazione CPU
    init_array(n, POLYBENCH_ARRAY(A));
    

    // Prefetch della matrice alla GPU  (per performance)
    cudaMemPrefetchAsync(A, size, 0);

    // ========== GPU ==========
    polybench_start_instruments;
    kernel_lu_cuda(n, (double*)POLYBENCH_ARRAY(A)); 
    polybench_stop_instruments;
    printf("Tempo GPU: ");
    polybench_print_instruments;

    // Prefetch alla CPU (per la validazione)
    cudaMemPrefetchAsync(A, size, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    // Liberazione UVM
    cudaFree(A);

    return 0;
}

#endif
