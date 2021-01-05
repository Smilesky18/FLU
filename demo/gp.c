# include <stdio.h>
# include <stdlib.h>
# include <float.h>
# include <string.h>
# include <immintrin.h>
# include <sys/time.h>
# include <math.h>
# include <omp.h>
# include <stdbool.h>

void gp(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U,double *L, double *U, double *xx)
{
    int i, j, k, current_column;
	int columns, column_end, val, column_start;
	double temp, U_diag;
	for ( k = 0; k < n; k++ )
	{

			  current_column = perm_c[k];
			  for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
			  {
				  xx[perm_r[row_ptr[j]]] = a[j];
			  }
			  columns = offset_U[k+1] - offset_U[k] - 1;
			  column_end = row_ptr_U[offset_U[k+1] - 2];

			  for ( j = 0; j < columns; j++ )
			  {
				  val = j+offset_U[k];
				  column_start = row_ptr_U[val];
				  temp = xx[column_start];

				  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
				  {
					  xx[row_ptr_L[i]] -=  temp*L[i];
                  }
			  }
			  
			  for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
			  {
				  U[i] = xx[row_ptr_U[i]];
				  xx[row_ptr_U[i]] = 0;
			  }
			  
			  U_diag = U[i-1];
			  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
			  {
				  L[i] = xx[row_ptr_L[i]] / U_diag;
				  xx[row_ptr_L[i]] = 0;
			  }
	 }
}

