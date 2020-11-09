# include <stdio.h>
# include <stdlib.h>
# include <float.h>
# include <string.h>
# include <immintrin.h>
# include <sys/time.h>
# include <math.h>
# include <omp.h>
# include <stdbool.h>

typedef union{
	unsigned int bit32;
	char boolvec[4];
} bitInt;

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

// int mmm_multi_value( int a, int b )
// {
// 	if ( a < b ) return a;
// 	return b;
// }

// double microtime()
// {
// 	struct timeval tv;
// 	struct timezone tz;
// 	gettimeofday(&tv,&tz);

// 	return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
// }

double* lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U, int *sn_record, int thresold, int sn_sum, int *flag, double *nic_x, int num_thread, int thread_number, int *start, int *end, double *L, double *U, double **xx1, double **xx2, double **dv1, double **dv2, int thread_number_prior_level, int *asub_U_level, double *lx, double *ux, bitInt *tag, int sum_level, int max_level, int *xa_trans)
{

	// printf("func: num_th = %d n/4+1 = %d\n", num_thread, n/4+1);

	// for ( int i = 0; i < (n/4+1); i++ )
	// {
	// 	printf("%d: %d %d %d %d\n", i , tag[i].boolvec[0], tag[i].boolvec[1], tag[i].boolvec[2], tag[i].boolvec[3]);
	// 	// printf("%d: %d\n", tag[i].bit32);
	// }
	// printf("next next next next next !\n");
   omp_set_num_threads(num_thread);
	// omp_set_num_threads(1);

  #pragma omp parallel
{
	double *xx_next_column;
	double U_diag;
    int j, k, current_column;
    int i;
	int  val, columns, column_end, pack, column_start, column_number_sn, column_sn_end, row_num, row_count, column_divid, divid, row_sn_start, qqq, row_else1, row_else2, dense_vec_counter, sss, dividd, row_record, current_next_column, columns_next_column, column_end_next_column, j_next_column, pack_next_column, val_next_column, column_start_next_column, column_number_sn_next_column, pack_k, rrr, row_column, row_column_start, pack_j, row_else3, row_else4;
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
	// int M;

	int div;

	#pragma omp for schedule(static, 1)
	  for ( m = thread_number_prior_level; m < thread_number; m++ )
	 {
		  {
			  for ( kk = start[m]; kk <= end[m]; kk++ )
			// #pragma omp for schedule(static, 1)
			//   for ( kk = xa_trans[thread_number_prior_level]; kk < xa_trans[thread_number_prior_level+1]; kk++ )
			  {
			//   printf("kk = %d k = %d start = %d end = %d\n", kk, asub_U_level[kk], start[m], end[m]);
			  k = asub_U_level[kk];
			//   k = kk;
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
						  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
						  {
							  xx_next_column[row_ptr_L[i]] -=  temp*L[i];
                          }
					  pack_j = 1;
				  }
				  else
				  {
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
						   
						  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
						  v.vec = _mm512_i32gather_pd(vi.vec, &xx_next_column[0], 8);
						  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
						  _mm512_i32scatter_pd(&xx_next_column[0], vi.vec, v_sub.vec, 8);

						  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
						  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx_next_column[0], 8);
						  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
						  _mm512_i32scatter_pd(&xx_next_column[0], vi_2.vec, v_sub_2.vec, 8); 
						  
						  dividd--;
					  }
					  
					  pack_j = column_number_sn;
				  }    
			  }
			  
			  // _mm_prefetch(&asub_U_level[kk+1], _MM_HINT_T0);
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
			//   printf("n/4+1 = %d k/4 = %d k mod 4 = %d k = %d\n", n/4+1, k/4, k%4, k);
			  tag[k/4].boolvec[k%4] = 1;
			  	//   printf("n/4+1 = %d k/4 = %d k mod 4 = %d k = %d tag = %d\n", n/4+1, k/4, k%4, k, tag[k/4].boolvec[k%4]);
			//   if ( k == 99336 ) printf("yeah! k = 99336\n");
		  }
		  }
		//  #pragma omp barrier
	 }
}


	// t2 = microtime() - t1;
	// printf("t2 = %lf\n", t2);
	//printf("sum_t = %.16lf\n", sum_t);
	//printf("row_record_counter = %d\n", row_record_counter);
	//printf("sum_flag_0 = %d sum_flag_1 = %d sum_flag_2 = %d\n", sum_flag_0, sum_flag_1, sum_flag_2);
  /* solve for Ly = b and Ux = y */
//    double *y, *x;
//    int ij, ji;
/*   printf("nzl = %d\n", nzl);
for ( ij = 0; ij < nzl; ij++ )
{
	//printf("nicslu_L[%d] = %lf\n", ij, lx[ij]);
	printf("nicslu_L[%d] = %lf me_L[%d] = %lf\n", ij, lx[ij], ij, L[ij]);
	//printf("nicslu_L[%d] = %lf me_L[%d] = %lf\n", ij, lx[ij], ij, L[ij]);
}*/

 /* y = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
  x = ( double *)_mm_malloc( sizeof( double ) * n, 64 );

  for ( ij = 0; ij < n; ij++ )
  {
	  y[ij] = 1.0;
  }

  for ( ij = 0; ij < n; ij++ )
  {
	  for ( ji = offset_L[ij]+1; ji < offset_L[ij+1]; ji++ )
	  {
		  y[row_ptr_L[ji]] -= y[ij] * L[ji];
	  }
  }

  //x[n-1] = y[n-1];
  for ( ij = 0; ij < n; ij++ )
  {
	  x[ij] = y[ij];
  }

  x[n-1] = y[n-1]/U[nzu-1];
  for ( ij = n-1; ij > 0; ij-- )
  {
	  for ( ji = offset_U[ij]; ji < offset_U[ij+1]-1; ji++ )
	  {
		  x[row_ptr_U[ji]] -= x[ij] *U[ji];
	  }
	  x[ij-1] = x[ij-1]/U[offset_U[ij]-1];
  }   
  
  double *x_real = (double *)malloc(sizeof(double) * n);
  int error_lu_gp = 0;
  int error_nic = 0;
  for ( ij = 0; ij < n; ij++ ) x_real[perm_c[ij]] = x[ij];
  for ( ij = 0; ij < n; ij++ )
  {
	  //printf("nic_x[%d] = %lf x[%d] = %lf\n", ij, nic_x[ij], ij, x_real[ij]);
	  if ( fabs(nic_x[ij]-x_real[ij]) > 0.1 )
	  {
		  error_nic++;
		  //printf("nicslu[%d] = %lf me[%d] = %lf\n", ij, nic_x[ij], ij, x_real[ij]);
	  }	
  }
  printf("error of Nic results are: %d\n", error_nic); 
  for ( ij = 0; ij < offset_L[prior_column]; ij++ )
  {
	  if ( fabs(lx[ij]-L[ij]) > 0.1 )
	  {
		  error_nic++;
		  printf("prior-nicslu_L[%d] = %lf me_L[%d] = %lf\n", ij, lx[ij], ij, L[ij]);
	  }	
  }*/

  return U;
}

