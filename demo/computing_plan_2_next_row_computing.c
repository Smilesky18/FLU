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



double* lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U, int *sn_record, int thresold, double *L, double *U, double **xx1, double **xx2, double **dv1, double **dv2, int *asub_U_level, double *lx, double *ux, char *tag, int max_level, int pri_level, int *xa_trans, int num_thread)
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
	int dense_vec;
	int start, end;
	double sum = 0;
	int pa;

	#pragma omp for schedule(static, 1)	
	for ( kk = xa_trans[0]; kk < xa_trans[max_level]; kk++ )
	// for ( k = 0; k < n; k++ )
	  {
		  {
			//   for ( kk = start[m]; kk <= end[m]; kk++ )
			//   for ( kk = xa_trans[m]; kk < xa_trans[m+1]; kk++ )
			  {
			  k = asub_U_level[kk];
			//   if ( tn == 0 ) 
			  	// printf("k = %d kk = %d tn = %d start = %d end = %d\n", k, kk, tn, xa_trans[sum_level], xa_trans[max_level]-1);
			// if ( k == 95273 || k == 95274 ) printf("func: k = %d tn = %d\n", k, tn);
			//   k = kk;
			  current_column = perm_c[k];
			  for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
			  {
				  xx_next_column[perm_r[row_ptr[j]]] = a[j];
				// xx_next_column[row_ptr[j]] = a[j];
			  }
			  columns = offset_U[k+1] - offset_U[k] - 1;
			  column_end = row_ptr_U[offset_U[k+1] - 2];

			  for ( j = 0; j < columns; j+=pack_j )
			  {
				  val = j+offset_U[k];
				  column_start = row_ptr_U[val];
				
				  temp = xx_next_column[column_start];
				  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
				  else column_number_sn = column_end-column_start+1;


				  if ( column_number_sn < 8)
				  {
					  		column_next = row_ptr_U[val];
							wait = (volatile char*)&(tag[column_next]);
							
							while ( !(*wait) ) 
							{ 
								;
							}
						  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
						  {
							  xx_next_column[row_ptr_L[i]] -=  temp*L[i];
                          }
					  pack_j = 1;
				  }
				  
				  else
				  {
					  for ( jj = 0; jj < column_number_sn; jj++ )
					  {
						  	column_next = row_ptr_U[val+jj];
							wait = (volatile char*)&(tag[column_next]);
							
							while ( !(*wait) ) 
							{ 
								;
							}
					  }
					 
					  column_sn_end = row_ptr_U[val+column_number_sn-1];
					  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
					  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
					  column_divid = row_num % 16;
					  divid = row_num / 16;
					  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
					  dense_vec = 0;

					  /*for ( qqq = 0; qqq < column_number_sn; qqq++ )
					  {
						  row_else1 = row_ptr_U[val+qqq];
						  temp = xx_next_column[row_else1];

						  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]; sss++ )
						  {
							  xx_next_column[row_ptr_L[sss]] -=  temp*L[sss];
						  }
					  }*/

					  for ( qqq = 0; qqq < column_number_sn; qqq++ )
					  {
						  row_else1 = row_ptr_U[val+qqq];
						  temp = xx_next_column[row_else1];
						  dense_vec_counter = qqq;

						  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-row_num; sss++ )
						  {
							  xx_next_column[row_ptr_L[sss]] -=  temp*L[sss];
						  }
						  for ( sss = offset_L[row_else1+1]-row_num; sss < offset_L[row_else1+1]; sss++ )
						  {
							  dense_vec_2[dense_vec_counter] = L[sss];
							  dense_vec_counter += column_number_sn;
						  }
					  }
					  dense_vec_counter = 0;
					  for ( sss = 0; sss < row_num; sss++ )
					  {
						//   #pragma unroll(4)
						  for ( qqq = 0; qqq < column_number_sn; qqq++ )
						  {
							  row_else1 = row_ptr_U[val+qqq];
							  temp = xx_next_column[row_else1];
							//   start = offset_L[row_else1+1]-row_num+sss;
							  //sum += temp * L[start];
							//   end = offset_L[row_else1+2]-row_num+sss;
							//   _mm_prefetch((char*)&L[end], _MM_HINT_T0);
							  sum += temp * dense_vec_2[dense_vec_counter++];
						  }
						  row_else2 = row_ptr_L[offset_L[column_sn_end]-row_num+sss];
						  xx_next_column[row_else2] -= sum;
						  sum = 0;
					  }

					  pack_j = column_number_sn;
				  }   
			  
			  }
			  
			  // _mm_prefetch(&asub_U_level[kk+1], _MM_HINT_T0);
			  for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
			  {
				  U[i] = xx_next_column[row_ptr_U[i]];
				//   if ( i == 1448307 ) printf("In next func: xx[%d] = %lf\n", row_ptr_U[i], xx_next_column[row_ptr_U[i]]);
				  xx_next_column[row_ptr_U[i]] = 0;
			  }
			  
			  U_diag = U[i-1];
			  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
			  {
				  L[i] = xx_next_column[row_ptr_L[i]] / U_diag;
				  xx_next_column[row_ptr_L[i]] = 0;
			  }
			//   tag[k/4].boolvec[k%4] = 1;
			tag[k] = 1;
			// if ( tn == 0 ) 
			//  printf("k = %d end tn = %d\n", k, tn);
		  }
		    
		  }
		  
	 }
}

  return U;
}

