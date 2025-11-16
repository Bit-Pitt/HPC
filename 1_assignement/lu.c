#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"

/* Array initialization. */
// Non accellerarlo perche' non e' quello che mi importa
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
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

#include <omp.h>

#ifdef GPU
#define NTHREAD_GPU 128
void kernel_lu(int n, DATA_TYPE A[n][n])
{
  int k, i, j;

  //Mappo una sola volta sia in entrata nella gpu che in uscita
  #pragma omp target data map(tofrom: A[0:n][0:n])
  {
    for (k = 0; k < n; ++k)
    {

      #pragma omp target teams distribute parallel for  num_teams((n + NTHREAD_GPU - 1)/NTHREAD_GPU) thread_limit(256)
      for (j = k + 1; j < n; ++j) 
        A[k][j] = A[k][j] / A[k][k];
      
    
      #pragma omp target teams distribute parallel  for collapse(2)  num_teams((n + NTHREAD_GPU - 1)/NTHREAD_GPU) thread_limit(256)
      for (i = k + 1; i < n; ++i) 
        for (j = k + 1; j < n; ++j) 
          A[i][j] = A[i][j] - A[i][k] * A[k][j];
        
    }
  } 
}
#endif

#ifdef CPU_TASKS
void kernel_lu(int n, DATA_TYPE A[n][n])
{
  int k, i, j;

    for (k = 0; k < n; ++k)
    {
	    //#pragma omp taskloop num_tasks(32) 
		#pragma omp task depend(out:A[0:n][0:n])
        for (j = k + 1; j < n; ++j) 
		  //#pragma omp task depend(out:A[0:n][0:n])
          A[k][j] = A[k][j] / A[k][k];
		
		//#pragma omp taskwait
	    //#pragma omp taskloop num_tasks(32) 
		#pragma omp task depend(in:A[0:n][0:n])
        for (i = k + 1; i < n; ++i) 
          for (j = k + 1; j < n; ++j) 
		    //#pragma omp task depend(in:A[0:n][0:n])
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
		
	}
}
#endif

#ifdef STATIC
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j, k;

  #pragma omp parallel  num_threads(4) private(i, j, k) shared(A,n)
  {
    for (k = 0; k < n; k++)
    {
      if (fabs(A[k][k]) < 0.00000000001) {
          printf("Pivot troppo piccolo\n");
      }
      #pragma omp for schedule(static) 
      for (j = k + 1; j < n; j++)
        A[k][j] = A[k][j] / A[k][k];

      #pragma omp for collapse(2) schedule(static) 
      for (i = k + 1; i < n; i++)
        for (j = k + 1; j < n; j++)
          A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
  }
}
#endif

#ifdef SEQ
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j, k;

  
    for (k = 0; k < n; k++)
    {
     
      for (j = k + 1; j < n; j++)
        A[k][j] = A[k][j] / A[k][k];

      
      for (i = k + 1; i < n; i++)
        for (j = k + 1; j < n; j++)
          A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
  
}
#endif

#ifdef DYN
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j, k;

  #pragma omp parallel  num_threads(4) private(i, j, k) shared(A,n)
  {
    for (k = 0; k < n; k++)
    {
      if (fabs(A[k][k]) < 0.00000000001) {
          printf("Pivot troppo piccolo\n");
      }
      #pragma omp for schedule(dynamic) 
      for (j = k + 1; j < n; j++)
        A[k][j] = A[k][j] / A[k][k];

      #pragma omp for collapse(2) schedule(dynamic) 
      for (i = k + 1; i < n; i++)
        for (j = k + 1; j < n; j++)
          A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
  }
}
#endif


#ifdef VAL    
static void kernel_lu_val(int n, double A[n][n])
{
  int k, i, j;        //copiare la funzione che si vuole validare

  //Mappo una sola volta sia in entrata nella gpu che in uscita
  #pragma omp target data map(tofrom: A[0:n][0:n])
  {
    for (k = 0; k < n; ++k)
    {

      #pragma omp target teams distribute parallel for thread_limit(256)
      for (j = k + 1; j < n; ++j) 
        A[k][j] = A[k][j] / A[k][k];
      
    
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
      for (i = k + 1; i < n; ++i) 
        for (j = k + 1; j < n; ++j) 
          A[i][j] = A[i][j] - A[i][k] * A[k][j];
        
    }
  } 
}
static void LU_sequenziale(int n, double A[n][n])
{
  int i, j, k;
  for (k = 0; k < n; k++) {
    
    for (j = k + 1; j < n; j++)
      A[k][j] = A[k][j] / A[k][k];

    for (i = k + 1; i < n; i++)
      for (j = k + 1; j < n; j++)
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
  }
}


static int confronta_matrici(int n, double A[n][n], double B[n][n], double tol)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (fabs(A[i][j] - B[i][j]) > tol)
        return 0; 
  return 1;
}


//Controllo di validazione, usiamo una matrice più piccola con valori deterministici per evitare problemi numerici
static void validazione(void)
{
  int n = 10;
  double A1[10][10];
  double A2[10][10];


  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      A1[i][j] = (float)(i*2 + j + 1);
      A2[i][j] = A1[i][j];
    }

  LU_sequenziale(n, A1);
  kernel_lu_val(n, A2);


  if (confronta_matrici(n, A1, A2, 0.001))
    printf("VALIDAZIONE PASSATA\n");
  else
    printf("VALIDAZIONE FALLITA\n");
}
#endif



// SEQ /  STATIC / DYN / VAL ...   ==> flag "-DSTATIC" per usare LU con parallelizzazione con static scheduling
int main(int argc, char **argv)
{

  // Flag definito a tempo di compilazione
  #ifdef VAL
  printf("Validazione della versione migliore trovata. (ignorare warning)\n");
  validazione();
  return 0;
  #endif
  


  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_lu(n, POLYBENCH_ARRAY(A));
  

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}








