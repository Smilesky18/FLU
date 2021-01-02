# include <stdio.h>
# include <stdlib.h>
# include <float.h>
# include <string.h>
# include <immintrin.h>
# include <sys/time.h>
# include <math.h>
# include <omp.h>
# include <stdbool.h>
#define MICRO_IN_SEC 1000000.00

  typedef __attribute__((aligned(64))) union
  {
    __m512d vec;
    double ptr_vec[8];
  }v2df_t;

  typedef union
  {
    __m256i vec;
    int ptr_vec[8];
  }v2if_t;

  typedef union{
	unsigned int bit32;
	char boolvec[4];
} bitInt;



double* flu_right_looking(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U, int *sn_record, int thresold, double *L, double *U, double **xx1, double **xx2, double **dv1, double **dv2, char *tag, int num_thread, int *row_ptr_U_trans, int *depend_L_col, int *len_U_in_computation, int *offset_L_plus)
{
   omp_set_num_threads(num_thread);

  #pragma omp parallel
{
	double *xx_next_column;
	double U_diag;
    int j, k, current_column;
    int i;
	int  val, columns, column_end, pack, column_start, column_number_sn, column_sn_end, row_num, row_count, column_divid, divid, row_sn_start, qqq, row_else1, row_else2, dense_vec_counter, sss, dividd, row_record, current_next_column, columns_next_column, column_end_next_column, j_next_column, pack_next_column, val_next_column, column_start_next_column, column_number_sn_next_column, pack_k, rrr, row_column, row_column_start, pack_j, column_next, row_else3, row_else4;
    double temp, temp_next_column, U_diag_next_column;
    double *dense_vec_2;
	v2df_t v_l, v_row, v_mul, v_sub, v, add_sum, zero, v_sub_2, v_2, add_sum_2, v_l_2, v_row_2, add_sum_3, add_sum_4, add_sum_5, add_sum_6, add_sum_7, add_sum_8, v_l_3, v_l_4, v_3, v_4, v_sub_3, v_sub_4, x0, x1, x2, x3;
    v2if_t vi, vi_2, vi_3, vi_4;
    zero.vec = _mm512_setzero_pd();
	int kk, m, columns_1, columns_next_column_1;
	
	int tn = omp_get_thread_num();
	xx_next_column = xx2[tn];
	dense_vec_2 = dv2[tn];
	volatile char *wait;

	int div;
	int wait_col;
	int val2;
	int jj;
	int dep_col;
	int len;

	#pragma omp for schedule(static, 1)	
	for ( kk = 0; kk < nzu; kk++ )
	{
			  k = row_ptr_U_trans[kk];
			  dep_col = depend_L_col[kk];
			  len = len_U_in_computation[kk];

			  if ( dep_col == -1 )
			  {
				  U_diag = U[offset_U[k+1] - 1];
				  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
				  {
						L[i] = L[i] / U_diag;
				  }
				  tag[k] = 1;
			  }
			  else
			  {
				for ( j = offset_U[k]+len; j < offset_U[k+1]; j++ )
				{
					xx_next_column[row_ptr_U[j]] = U[j];
				}
				for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
				{
					xx_next_column[row_ptr_L[j]] = L[j];
				}
				wait = (volatile char*)&(tag[dep_col]);	
				while ( !(*wait) ) 
				{ 
					;
				}
				temp = xx_next_column[dep_col];
				for ( i = offset_L[dep_col]+1; i < offset_L[dep_col+1]; i++ )
				{
					xx_next_column[row_ptr_L[i]] -=  temp*L[i];
				}

				for ( i = offset_U[k]+len; i < offset_U[k+1]; i++ )
				{
					U[i] = xx_next_column[row_ptr_U[i]];
					xx_next_column[row_ptr_U[i]] = 0;
				}
				
				// U_diag = U[i-1];
				for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
				{
					L[i] = xx_next_column[row_ptr_L[i]];
					xx_next_column[row_ptr_L[i]] = 0;
				}
			  }  
	}
}

  return U;
}

