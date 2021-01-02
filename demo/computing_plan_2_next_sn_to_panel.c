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

double microtime_func()
{
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

int dfs(int *asub, int *xa, int *be, int k, int *offset_U, int *row_ptr_U, int *value, int n, int *temp)
{
	int i, j;
	int sum = 0;
	xa[0] = 0;

	for ( i = k; i < k+8; i++ )
	{
		for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
		{
			value[row_ptr_U[j]]++;
		}
	}
	for ( i = 0; i < n; i++ )
	{
		if ( value[i] )
		{
			asub[sum] = i;
			temp[i] = sum;
			sum++;
			xa[sum] += value[i];
		}
	}
	for ( i = k; i < k+8; i++ )
	{
		for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
		{
			be[temp[row_ptr_U[j]] 
		}
	}

}

double* lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_panel(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U, int *sn_record, int thresold, double *L, double *U, double **xx1, double **xx2, double **dv1, double **dv2, int *asub_U_level, double *lx, double *ux, char *tag, int max_level, int pri_level, int *xa_trans, int num_thread, int *sn_number, int *sn_column_start, int *sn_column_end, int *xa, int *asub, int *offset_U_row, int *col_ptr_U)
{
   omp_set_num_threads(num_thread);

   	double *sum_sn = (double *)malloc(sizeof(double) * n);
	memset(sum_sn, 0, sizeof(double) * n);

	int *sum_sn_num = (int *)malloc(sizeof(int) * n);
	memset(sum_sn_num, 0, sizeof(int) * n);

	double *xx = (double *)malloc(sizeof(double) * 8*n);
	memset(xx, 0, sizeof(double) * 8*n);

	double t_sum = 0;

  #pragma omp parallel
{
	double *xx_next_column;
	double U_diag;
    int j, k, current_column;
    int i;
	int  val, columns, column_end, pack, column_start, column_number_sn, column_sn_end, row_num, row_count, column_divid, divid, row_sn_start, qqq, row_else1, row_else2, dense_vec_counter, sss, dividd, row_record, row_record_2, current_next_column, columns_next_column, column_end_next_column, j_next_column, pack_next_column, val_next_column, column_start_next_column, column_number_sn_next_column, pack_k, rrr, row_column, row_column_start, pack_j, column_next, row_else3, row_else4;
    double temp, temp_next_column, U_diag_next_column;
    double *dense_vec_2;
	v2df_t v_l, v_row, v_mul, v_sub, v, add_sum, zero, v_sub_2, v_2, add_sum_2, v_l_2, v_row_2, add_sum_3, add_sum_4, add_sum_5, add_sum_6, add_sum_7, add_sum_8, v_l_3, v_l_4, v_3, v_4, v_sub_3, v_sub_4, x0, x1, x2, x3;
    v2if_t vi, vi_2, vi_3, vi_4;
    zero.vec = _mm512_setzero_pd();
	int kk, m, columns_1, columns_next_column_1;
	
	int tn = omp_get_thread_num();
	// int tn = 0;
	xx_next_column = xx2[tn];
	dense_vec_2 = dv2[tn];
	volatile char *wait;

	int div;
	int wait_col;
	int val2;
	int jj; 
	int sum;
	int ssss;
	double t1, t2;
	int sn_rows, ii, col, iii;

	#pragma omp for schedule(static, 1)	
	// for ( kk = xa_trans[pri_level]; kk < xa_trans[max_level]; kk++ )
	for ( k = 0; k < n; k++ )
	  {
		//   printf("tn = %d\n", tn);
		  {
			//   for ( kk = start[m]; kk <= end[m]; kk++ )
			//   for ( kk = xa_trans[m]; kk < xa_trans[m+1]; kk++ )
			  {
				//   t1 = microtime_func();
			//   k = asub_U_level[kk];
			//   if ( tn == 0 ) 
			  	// printf("k = %d kk = %d tn = %d start = %d end = %d\n", k, kk, tn, xa_trans[sum_level], xa_trans[max_level]-1);
			// if ( k == 95273 || k == 95274 ) printf("func: k = %d tn = %d\n", k, tn);
			//   k = kk;
			if ( sn_number[k] >= 0 )
			{
				for ( i = 0; i < 8; i++ )
				{
					current_column = perm_c[k+i];
					for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
					{
						xx[perm_r[row_ptr[j]]+i*n] = a[j];
					}
				}
			
				for ( ii = xa[sn_number[k]]; ii < xa[sn_number[k]+1]; ii++ )
				{
					jj = asub[ii]; //jj是要用来更新列的行号
					for ( iii = offset_U_row[jj]; iii < offset_U_row[jj+1]; iii++ )
					{
						col = col_ptr_U[iii];

						temp = xx[jj+col*n];
						for ( i = offset_L[jj]+1; i < offset_L[jj+1]; i++ )
						  {
							  xx[row_ptr_L[i]+col*n] -=  temp*L[i];
                          }
						offset_U_row[jj]++;
					}
				}
				
			}
              
			  current_column = perm_c[k];
			  for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
			  {
				  xx_next_column[perm_r[row_ptr[j]]] = a[j];
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

				  if ( column_number_sn < thresold )
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
					  div = column_number_sn % 4;

				      for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
					  {
						  row_else1 = row_ptr_U[val+qqq];
						  row_else2 = row_ptr_U[val+qqq+1];
						  temp = xx_next_column[row_else1];
						  dense_vec_counter = qqq;
						  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
						  {
							  dense_vec_2[dense_vec_counter++] += temp*L[sss];
						  }
						  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start-1];
						  dense_vec_2[row_else2-column_start-1] = 0;
					  }
					  temp = xx_next_column[column_sn_end];
					  dense_vec_counter = column_number_sn - 1;
					  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
					  {
						  dense_vec_2[dense_vec_counter++] += temp*L[i];
					  }
					  dense_vec_counter = column_number_sn - 1;
					  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
					  {
						  xx_next_column[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
						  dense_vec_2[dense_vec_counter] = 0;
						  dense_vec_counter++;
					  }
					  
					  dividd = divid;
					  for ( qqq = 0; qqq < divid; qqq++ )
					  {
						  add_sum.vec = zero.vec; 
						  add_sum_2.vec = zero.vec;
				
						  for ( sss = 0; sss < column_number_sn; sss++ )
						  {
							  row_else1 = row_ptr_U[val+sss];
							  row_record = offset_L[row_else1+1] - dividd*16;
							  
							  v_row.vec = _mm512_set1_pd(xx_next_column[row_else1]);
							  v_l.vec = _mm512_load_pd(&L[row_record]);
							  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

							  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
							  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
						  }
						   
						  vi.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
						  v.vec = _mm512_i32gather_pd(vi.vec, &xx_next_column[0], 8);
						  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
						  _mm512_i32scatter_pd(&xx_next_column[0], vi.vec, v_sub.vec, 8);

						  vi_2.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
						  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx_next_column[0], 8);
						  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
						  _mm512_i32scatter_pd(&xx_next_column[0], vi_2.vec, v_sub_2.vec, 8); 
						  
						  dividd--;
					  }
					
					  pack_j = column_number_sn;
				  }   
			  }
			  
			  for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
			  {
				  U[i] = xx_next_column[row_ptr_U[i]];
				  xx_next_column[row_ptr_U[i]] = 0;
			  }
			  
			  U_diag = U[i-1];
			  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
			  {
				  L[i] = xx_next_column[row_ptr_L[i]] / U_diag;
				  xx_next_column[row_ptr_L[i]] = 0;
			  }
			tag[k] = 1;
		  }
		    
		  }
		
	 }
	//    #pragma omp barrier 
}

// for ( int i = 0; i < n; i++ )
// {
// 	if ( sum_sn[i] > 0 )
// 		printf("%d %d %lf\n", i, offset_U[i+1]-offset_U[i], sum_sn[i]);
// }

// for ( int i = 0; i < n; i++ )
// {
// 	if ( sum_sn_num[i] > 0 )
// 		printf("%d %d %d\n", i, offset_U[i+1]-offset_U[i], sum_sn_num[i]);
// }

// printf("single columns computing time is: %lf\n", t_sum);

  return U;
}

