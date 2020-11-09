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

double* lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_next(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U, int *sn_record, int thresold, int *sn_num_record, int *sn_column_start, int *sn_column_end, int sn_sum, int *flag, double *nic_x, int prior_column, int next_column, int num_thread, int thread_number, int *start, int *end, int *seperator_in_column, double *L, double *U, double **xx1, double **xx2, double **dv1, double **dv2, int thread_number_prior_level, int *asub_U_level, int *seperator_in_column_2, int *sign, double *lx, double *ux, int thread_number_c, int thread_number_a, bitInt *tag, int *assign)
{
  omp_set_num_threads(num_thread);

  int ii;
 #pragma omp parallel
{
	double *xx, *xx_next_column;
	double U_diag;
    int j, k, current_column;
    int i;
	int  val, columns, column_end, pack, column_start, column_number_sn, column_sn_end, row_num, row_count, column_divid, divid, row_sn_start, qqq, row_else1, row_else2, dense_vec_counter, sss, dividd, row_record, current_next_column, columns_next_column, column_end_next_column, j_next_column, pack_next_column, val_next_column, column_start_next_column, column_number_sn_next_column, pack_k, rrr, row_column, row_column_start, pack_j;
    double temp, temp_next_column, U_diag_next_column;
    double *dense_vec, *dense_vec_2;
	v2df_t v_l, v_row, v_mul, v_sub, v, add_sum, zero, v_sub_2, v_2, add_sum_2, v_l_2, v_row_2, add_sum_3, add_sum_4, add_sum_5, add_sum_6, add_sum_7, add_sum_8, v_l_3, v_l_4, v_3, v_4, v_sub_3, v_sub_4;
    v2if_t vi, vi_2, vi_3, vi_4;
    zero.vec = _mm512_setzero_pd();
	int kk, m, columns_1, columns_next_column_1, M, D, thr_num;
	int tn = omp_get_thread_num();
	// int tn = 0;
	xx = xx1[tn];
	xx_next_column = xx2[tn];
	dense_vec = dv1[tn];
	dense_vec_2 = dv2[tn];
	thr_num = tn;
   {
	//  #pragma omp for //schedule(static, 1)
//	for (  thr_num = 0; thr_num < 16; thr_num++ )
	{
		// xx = xx1[thr_num];
		// xx_next_column = xx2[thr_num];
		// dense_vec = dv1[thr_num];
		// dense_vec_2 = dv2[thr_num];

	  for (  M = 0; M <= thread_number_prior_level; M++ )
	  {
		  m = M*num_thread+thr_num+thread_number_c;
		//   m = M;
		  {  
		  if ( sign[m] == 0 ) // parallel < serial : 
		  {
		   for ( k = start[m]; k <= end[m]; k+=pack_k )
		  {
			//   printf("k = %d n = %d m = %d M = %d\n", k, n, m, M);
			   if ( flag[k] == 2 )
			   {
				//    printf("flag == 2\n");
				    //  current_column = perm_c[k];
					//  current_next_column = perm_c[k+1];
					// for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
					// {
					// 	xx[perm_r[row_ptr[j]]] = a[j];
					// }
					// for ( j = offset[current_next_column]; j < offset[current_next_column+1]; j++ )
					// {
					// 	xx_next_column[perm_r[row_ptr[j]]] = a[j];
					// }
					  for ( j = offset_U[k]+seperator_in_column_2[k]; j < offset_U[k+1]; j++ )
					  {
						  xx[row_ptr_U[j]] = U[j];
					  }
					  for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
					  {
						  xx[row_ptr_L[j]] = L[j];
					  }
					  for ( j = offset_U[k+1]+seperator_in_column_2[k+1]; j < offset_U[k+2]; j++ )
					  {
						  xx_next_column[row_ptr_U[j]] = U[j];
					  }
					  for ( j = offset_L[k+1]+1; j < offset_L[k+2]; j++ )
					  {
						  xx_next_column[row_ptr_L[j]] = L[j];
					  }
	
					  columns_1 = seperator_in_column_2[k];
					  columns = offset_U[k+1] - offset_U[k] - 1;
					  column_end = row_ptr_U[offset_U[k] + columns - 1];

					  columns_next_column_1 = seperator_in_column_2[k+1];
					  columns_next_column = offset_U[k+2] - offset_U[k+1] - 1;
					  column_end_next_column = row_ptr_U[offset_U[k+1] + columns_next_column-1];
					  
					 for ( j = columns_1, j_next_column = columns_next_column_1; j < columns || j_next_column < columns_next_column; j+=pack, j_next_column+=pack_next_column )
					  {
						  val = j+offset_U[k];
						  val_next_column = j_next_column+offset_U[k+1];

						  column_start = row_ptr_U[val];
						  column_start_next_column = row_ptr_U[val_next_column];

						  temp = xx[column_start];
						  temp_next_column = xx_next_column[column_start_next_column];

						  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
						  else column_number_sn = column_end-column_start+1;

						  if ( column_end_next_column-column_start_next_column > sn_record[column_start_next_column] ) column_number_sn_next_column = sn_record[column_start_next_column]+1;
						  else column_number_sn_next_column = column_end_next_column-column_start_next_column+1;
						  
						  if ( column_start < column_start_next_column )
						  {
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
								  }
								  pack = 1;
								  pack_next_column = 0;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
				
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter++] += temp*L[sss];
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
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
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  dividd--;
								  }
								  pack = column_number_sn;
								  pack_next_column = 0;
							  }
						  }
						  else if ( column_start > column_start_next_column )
						  {
							  if ( column_number_sn_next_column < thresold )
							  {
								  for ( i = offset_L[column_start_next_column]+1; i < offset_L[column_start_next_column+1]; i++ )
								  {
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 0;
								  pack_next_column = 1;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val_next_column+column_number_sn_next_column-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start_next_column + 1] - offset_L[column_start_next_column] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
								  
								  for ( qqq = 0; qqq < column_number_sn_next_column-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val_next_column+qqq];
									  row_else2 = row_ptr_U[val_next_column+qqq+1];
									  temp = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec_2[dense_vec_counter++] += temp*L[sss];
									  }
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start_next_column-1];
									  dense_vec_2[row_else2-column_start_next_column-1] = 0;
								  }
								  temp = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec_2[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( sss = offset_L[column_start_next_column+1]-row_num; sss < offset_L[column_start_next_column+1]-divid*16; sss++ )
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
									  for ( sss = 0; sss < column_number_sn_next_column; sss++ )
									  {
										  row_else1 = row_ptr_U[val_next_column+sss];
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
								  
								  pack = 0;
								  pack_next_column = column_number_sn_next_column;
							  }
						  }
						  else
						  {
			
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 1;
								  pack_next_column = 1;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
								  
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  temp_next_column = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter] += temp*L[sss];
										  dense_vec_2[dense_vec_counter] += temp_next_column*L[sss];
										  dense_vec_counter++;
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
									  
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start-1];
									  dense_vec_2[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  temp_next_column = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter] += temp*L[i];
									  dense_vec_2[dense_vec_counter] += temp_next_column*L[i];
									  dense_vec_counter++;
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
									  xx_next_column[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
									  dense_vec_2[dense_vec_counter] = 0;
									  dense_vec_counter++;
								  }
								  dividd = divid;
								  for ( qqq = 0; qqq < divid; qqq++ )
								  {
									  add_sum.vec = zero.vec; 
									  add_sum_2.vec = zero.vec;
									  add_sum_3.vec = zero.vec; 
									  add_sum_4.vec = zero.vec;
									  
									  for ( sss = 0; sss < column_number_sn; sss++ )
									  {
										  row_else1 = row_ptr_U[val+sss];
										  row_record = offset_L[row_else1+1] - dividd*16;
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_row_2.vec = _mm512_set1_pd(xx_next_column[row_else1]);
										  
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
										  
										  add_sum_3.vec = _mm512_fmadd_pd(v_row_2.vec, v_l.vec, add_sum_3.vec);	
										  add_sum_4.vec = _mm512_fmadd_pd(v_row_2.vec, v_l_2.vec, add_sum_4.vec);								  
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx_next_column[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum_3.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx_next_column[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_4.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi_2.vec, v_sub_2.vec, 8);
				 
									  dividd--;
								  }
								 
								  pack = column_number_sn;
								  pack_next_column = column_number_sn;
							  }
						  }	  
					  } 
					  
					  for ( i = offset_U[k]+seperator_in_column_2[k]; i < offset_U[k+1]; i++ )
					  {
						  U[i] = xx[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 0/1: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx[row_ptr_U[i]] = 0;
					  }
					  
					  U_diag = U[i-1];
					  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					  {
						  L[i] = xx[row_ptr_L[i]] / U_diag;
						//   if ( i == 17051593 ) printf("sign = 0/1: L[17051593] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						  xx[row_ptr_L[i]] = 0;
					  }
					  
					  for ( i = offset_U[k+1]+seperator_in_column_2[k+1]; i < offset_U[k+2]; i++ )
					  {
						  U[i] = xx_next_column[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 0/2: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx_next_column[row_ptr_U[i]] = 0;
					  }
					  
					  U_diag_next_column = U[i-1];
					  for ( i = offset_L[k+1]+1; i < offset_L[k+2]; i++ )
					  {
						  L[i] = xx_next_column[row_ptr_L[i]] / U_diag_next_column;
						//   if ( i == 17051593 ) printf("sign = 0/2: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx_next_column[row_ptr_L[i]], U_diag_next_column, k);
						  xx_next_column[row_ptr_L[i]] = 0;
					  }
					  pack_k = 2;
			   }
			   
			   else if ( flag[k] == 1 )
			   {
				      for ( j = offset_U[k]; j < offset_U[k+1]; j++ )
					  {
						  xx[row_ptr_U[j]] = U[j];
					  }
					  for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
					  {
						  xx[row_ptr_L[j]] = L[j];
					  }
					  for ( j = offset_U[k+1]; j < offset_U[k+2]; j++ )
					  {
						  xx_next_column[row_ptr_U[j]] = U[j];
					  }
					  for ( j = offset_L[k+1]+1; j < offset_L[k+2]; j++ )
					  {
						  xx_next_column[row_ptr_L[j]] = L[j];
					  }
	
					  columns_1 = seperator_in_column_2[k];
					  columns = offset_U[k+1] - offset_U[k] - 1;
					  column_end = row_ptr_U[offset_U[k] + columns - 1];

					  columns_next_column_1 = seperator_in_column_2[k+1];
					  columns_next_column = offset_U[k+2] - offset_U[k+1] - 1;
					  column_end_next_column = row_ptr_U[offset_U[k+1] + columns_next_column-1];
					  
					  for ( j = columns_1, j_next_column = columns_next_column_1; j < columns || j_next_column < columns_next_column-1; j+=pack, j_next_column+=pack_next_column )
					  {
						  val = j+offset_U[k];
						  val_next_column = j_next_column+offset_U[k+1];

						  column_start = row_ptr_U[val];
						  column_start_next_column = row_ptr_U[val_next_column];

						  temp = xx[column_start];
						  temp_next_column = xx_next_column[column_start_next_column];

						  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
						  else column_number_sn = column_end-column_start+1;

						  if ( column_end_next_column-column_start_next_column > sn_record[column_start_next_column] ) column_number_sn_next_column = sn_record[column_start_next_column]+1;
						  else column_number_sn_next_column = column_end_next_column-column_start_next_column+1;
						  
						  if ( column_start < column_start_next_column )
						  {
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
								  }
								  pack = 1;
								  pack_next_column = 0;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
					
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter++] += temp*L[sss];
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
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
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  dividd--;
								  }
								  
								  pack = column_number_sn;
								  pack_next_column = 0;
							  }
						  }
						  else if ( column_start > column_start_next_column )
						  {
							 if ( column_number_sn_next_column < thresold )
							 {
								  for ( i = offset_L[column_start_next_column]+1; i < offset_L[column_start_next_column+1]; i++ )
								  {
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 0;
								  pack_next_column = 1;
							  }
							  else
							  {
								   // printf("sn in tri!\n");
								  column_sn_end = row_ptr_U[val_next_column+column_number_sn_next_column-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start_next_column + 1] - offset_L[column_start_next_column] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
				
								  for ( qqq = 0; qqq < column_number_sn_next_column-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val_next_column+qqq];
									  row_else2 = row_ptr_U[val_next_column+qqq+1];
									  temp = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec_2[dense_vec_counter++] += temp*L[sss];
									  }
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start_next_column-1];
									  dense_vec_2[row_else2-column_start_next_column-1] = 0;
								  }
								  temp = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec_2[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( sss = offset_L[column_start_next_column+1]-row_num; sss < offset_L[column_start_next_column+1]-divid*16; sss++ )
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
									  for ( sss = 0; sss < column_number_sn_next_column; sss++ )
									  {
										  row_else1 = row_ptr_U[val_next_column+sss];
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
								  
								  pack = 0;
								  pack_next_column = column_number_sn_next_column;
							  }  
						  }
						  else
						  {
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 1;
								  pack_next_column = 1;
							  }
							  else
							  {
								   // printf("sn in tri!\n");
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;

								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  temp_next_column = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter] += temp*L[sss];
										  dense_vec_2[dense_vec_counter] += temp_next_column*L[sss];
										  dense_vec_counter++;
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
									  
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start-1];
									  dense_vec_2[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  temp_next_column = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter] += temp*L[i];
									  dense_vec_2[dense_vec_counter] += temp_next_column*L[i];
									  dense_vec_counter++;
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
									  xx_next_column[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
									  dense_vec_2[dense_vec_counter] = 0;
									  dense_vec_counter++;
								  }
								  dividd = divid;
								  for ( qqq = 0; qqq < divid; qqq++ )
								  {
									  add_sum.vec = zero.vec; 
									  add_sum_2.vec = zero.vec;
									  add_sum_3.vec = zero.vec; 
									  add_sum_4.vec = zero.vec;

									  for ( sss = 0; sss < column_number_sn; sss++ )
									  {
										  row_else1 = row_ptr_U[val+sss];
										  row_record = offset_L[row_else1+1] - dividd*16;
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_row_2.vec = _mm512_set1_pd(xx_next_column[row_else1]);
										  
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);
										 
										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
										  
										  add_sum_3.vec = _mm512_fmadd_pd(v_row_2.vec, v_l.vec, add_sum_3.vec);	
										  add_sum_4.vec = _mm512_fmadd_pd(v_row_2.vec, v_l_2.vec, add_sum_4.vec);								  
									  }
						
									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);
									  
									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx_next_column[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum_3.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx_next_column[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_4.vec);
									   _mm512_i32scatter_pd(&xx_next_column[0], vi_2.vec, v_sub_2.vec, 8);		
									  
									  dividd--;
								  }
								  pack = column_number_sn;
								  pack_next_column = column_number_sn; 
							  }  
						  }	  
					  } 
					  
					  for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
					  {
						  U[i] = xx[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 0/3: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx[row_ptr_U[i]] = 0;
					  }
					  
					  temp_next_column = xx_next_column[row_ptr_U[offset_U[k+2]-2]];
					  //temp_next_column = xx_next_column[k];
					  U_diag = U[i-1];
					  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					  {
						  L[i] = xx[row_ptr_L[i]] / U_diag;
						//   if ( i == 17051593 ) printf("sign = 0/3: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						  xx[row_ptr_L[i]] = 0;
						  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
					  }
					  
					  for ( i = offset_U[k+1]; i < offset_U[k+2]; i++ )
					  {
						  U[i] = xx_next_column[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 0/4: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx_next_column[row_ptr_U[i]] = 0;
					  }
					  
					  U_diag_next_column = U[i-1];
					  for ( i = offset_L[k+1]+1; i < offset_L[k+2]; i++ )
					  {
						  L[i] = xx_next_column[row_ptr_L[i]] / U_diag_next_column;
						//   if ( i == 17051593 ) printf("sign = 0/4: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx_next_column[row_ptr_L[i]], U_diag_next_column, k);
						  xx_next_column[row_ptr_L[i]] = 0;
					  } 
					  
					  pack_k = 2;
			   }
			
			    else
			   {
				  	for ( j = offset_U[k]; j < offset_U[k+1]; j++ )
					{
					  xx[row_ptr_U[j]] = U[j];
					}
					for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
					{
					  xx[row_ptr_L[j]] = L[j];
					} 

					// if ( k == 1089597 ) printf("sign = 0/5: xx[1089597] = %lf\n", xx[1089597]);
					// if ( k == 1089582 ) printf("sign = 0/5: xx[1089586] = %lf\n", xx[1089586]);

					/*if ( assign[m] )
					{
							current_column = perm_c[k];
							for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
							{
								xx[perm_r[row_ptr[j]]] = a[j];
							}
					}*/
				
				    // if ( k == 1089581 ) printf("sign = 0: xx[1089581] = %lf\n", xx[1089581]);

					row_column = offset_U[k+1] - offset_U[k] - 1;
					row_column_start = seperator_in_column_2[k];
					column_end = row_ptr_U[offset_U[k+1] - 2];
				   
				  	for ( j = row_column_start; j < row_column; j+=pack )
					{
					  val = j+offset_U[k];
					  column_start = row_ptr_U[val];
					  temp = xx[column_start];
					  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
					  else column_number_sn = column_end-column_start+1;

					  if ( column_number_sn < thresold )
					  {
							  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
							  {
								  xx[row_ptr_L[i]] -=  temp*L[i];
								//   if ( k == 1089597 && row_ptr_L[i] == 1089597 ) printf("sign = 0/5-compute: xx[1089597] = %lf L[%d] = %lf temp = %lf col_L = %d\n", xx[1089597], i, L[i], temp, column_start);
							  }
						  pack = 1;
					  }
					  else
					  {
						   // printf("sn in tri!\n");
						  column_sn_end = row_ptr_U[val+column_number_sn-1];
						  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
						  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
						  column_divid = row_num % 16;
						  divid = row_num / 16;
						  row_sn_start = offset_L[column_sn_end] + column_divid + 1;

						  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
						  {
							  row_else1 = row_ptr_U[val+qqq];
							  row_else2 = row_ptr_U[val+qqq+1];
							  temp = xx[row_else1];
							  dense_vec_counter = qqq;
							  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
							  {
								  dense_vec[dense_vec_counter++] += temp*L[sss];
							  }
							  xx[row_else2] -= dense_vec[row_else2-column_start-1];
							  dense_vec[row_else2-column_start-1] = 0;
						  }
						  temp = xx[column_sn_end];
						  dense_vec_counter = column_number_sn - 1;
						  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
						  {
							  dense_vec[dense_vec_counter++] += temp*L[i];
						  }
						  dense_vec_counter = column_number_sn - 1;
						  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
						  {
							  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
							  dense_vec[dense_vec_counter] = 0;
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
								  
								  v_row.vec = _mm512_set1_pd(xx[row_else1]);
								  v_l.vec = _mm512_load_pd(&L[row_record]);
								  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

								  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
								  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
							  }
							   
							  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
							  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
							  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
							  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

							  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
							  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
							  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
							  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
							  
							  dividd--;
						  }
						  pack = column_number_sn;
					  }    
				  }
				    
					for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
					{
						U[i] = xx[row_ptr_U[i]];
						// if ( i == 11564116 ) printf("sign = 0/5: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						// if ( i == 11564061 ) printf("sign = 0: k = %d xx[%d] = %lf \n", k, row_ptr_U[i], xx[row_ptr_U[i]]);
						xx[row_ptr_U[i]] = 0;
					}

					U_diag = U[i-1];
					for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					{
						L[i] = xx[row_ptr_L[i]] / U_diag;
						// if ( i == 17051593 ) printf("sign = 0/5: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						xx[row_ptr_L[i]] = 0;
					}  
					
					pack_k = 1;
			  }
		  }
		   
		  }

		  else if ( sign[m] == 1 ) // parallel > serial : 5~6
		  {
		 	for ( k = start[m]; k <= end[m]; k+=pack_k )
			{
			   if ( flag[k] )
			   {
					  current_column = perm_c[k];
					  current_next_column = perm_c[k+1];
					  for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
					  {
						  xx[perm_r[row_ptr[j]]] = a[j];
					  }
					  for ( j = offset[current_next_column]; j < offset[current_next_column+1]; j++ )
					  {
						  xx_next_column[perm_r[row_ptr[j]]] = a[j];
					  }
				
					  columns = seperator_in_column[k];
					  column_end = row_ptr_U[offset_U[k] + columns - 1];

					  columns_next_column = seperator_in_column[k+1];
					  column_end_next_column = row_ptr_U[offset_U[k+1] + columns_next_column-1];
					  
					  for ( j = 0, j_next_column = 0; j < columns || j_next_column < columns_next_column; j+=pack, j_next_column+=pack_next_column )
					  {
						  val = j+offset_U[k];
						  val_next_column = j_next_column+offset_U[k+1];

						  column_start = row_ptr_U[val];
						  column_start_next_column = row_ptr_U[val_next_column];

						  temp = xx[column_start];
						  temp_next_column = xx_next_column[column_start_next_column];

						  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
						  else column_number_sn = column_end-column_start+1;

						  if ( column_end_next_column-column_start_next_column > sn_record[column_start_next_column] ) column_number_sn_next_column = sn_record[column_start_next_column]+1;
						  else column_number_sn_next_column = column_end_next_column-column_start_next_column+1;
						  
						  if ( column_start < column_start_next_column )
						  {
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
								  }
								  pack = 1;
								  pack_next_column = 0;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
				
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter++] += temp*L[sss];
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
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
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  dividd--;
								  }
								  pack = column_number_sn;
								  pack_next_column = 0;
							  }
						  
						  }
						  
						  else if ( column_start > column_start_next_column )
						  {
							  if ( column_number_sn_next_column < thresold )
							  {
								  for ( i = offset_L[column_start_next_column]+1; i < offset_L[column_start_next_column+1]; i++ )
								  {
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 0;
								  pack_next_column = 1;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val_next_column+column_number_sn_next_column-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start_next_column + 1] - offset_L[column_start_next_column] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
								  
								  for ( qqq = 0; qqq < column_number_sn_next_column-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val_next_column+qqq];
									  row_else2 = row_ptr_U[val_next_column+qqq+1];
									  temp = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec_2[dense_vec_counter++] += temp*L[sss];
									  }
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start_next_column-1];
									  dense_vec_2[row_else2-column_start_next_column-1] = 0;
								  }
								  temp = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec_2[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( sss = offset_L[column_start_next_column+1]-row_num; sss < offset_L[column_start_next_column+1]-divid*16; sss++ )
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
									  for ( sss = 0; sss < column_number_sn_next_column; sss++ )
									  {
										  row_else1 = row_ptr_U[val_next_column+sss];
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
								  
								  pack = 0;
								  pack_next_column = column_number_sn_next_column;
							  }
						  
						  }
						  
						  else
						  {
			
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 1;
								  pack_next_column = 1;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
								  
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  temp_next_column = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter] += temp*L[sss];
										  dense_vec_2[dense_vec_counter] += temp_next_column*L[sss];
										  dense_vec_counter++;
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
									  
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start-1];
									  dense_vec_2[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  temp_next_column = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter] += temp*L[i];
									  dense_vec_2[dense_vec_counter] += temp_next_column*L[i];
									  dense_vec_counter++;
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
									  xx_next_column[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
									  dense_vec_2[dense_vec_counter] = 0;
									  dense_vec_counter++;
								  }
								  dividd = divid;
								  for ( qqq = 0; qqq < divid; qqq++ )
								  {
									  add_sum.vec = zero.vec; 
									  add_sum_2.vec = zero.vec;
									  add_sum_3.vec = zero.vec; 
									  add_sum_4.vec = zero.vec;
									  
									  for ( sss = 0; sss < column_number_sn; sss++ )
									  {
										  row_else1 = row_ptr_U[val+sss];
										  row_record = offset_L[row_else1+1] - dividd*16;
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_row_2.vec = _mm512_set1_pd(xx_next_column[row_else1]);
										  
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
										  
										  add_sum_3.vec = _mm512_fmadd_pd(v_row_2.vec, v_l.vec, add_sum_3.vec);	
										  add_sum_4.vec = _mm512_fmadd_pd(v_row_2.vec, v_l_2.vec, add_sum_4.vec);								  
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx_next_column[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum_3.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx_next_column[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_4.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi_2.vec, v_sub_2.vec, 8);
				 
									  dividd--;
								  }
								 
								  pack = column_number_sn;
								  pack_next_column = column_number_sn;
							  }
						  
						  }	  
					    
					  } 
					  
					  
					  for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
					  {
						  U[i] = xx[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 1/1: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx[row_ptr_U[i]] = 0;
					  }
					  
					  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					  {
						  //L[i] = xx[row_ptr_L[i]] / U_diag;
						  L[i] = xx[row_ptr_L[i]];
						//   if ( i == 17051593 ) printf("sign = 1/1: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						  xx[row_ptr_L[i]] = 0;
					  }
					  
					  for ( i = offset_U[k+1]; i < offset_U[k+2]; i++ )
					  {
						  U[i] = xx_next_column[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 1/2: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx_next_column[row_ptr_U[i]] = 0;
					  }
					  
					  //U_diag_next_column = U[i-1];
					  for ( i = offset_L[k+1]+1; i < offset_L[k+2]; i++ )
					  {
						  //L[i] = xx_next_column[row_ptr_L[i]] / U_diag_next_column;
						  L[i] = xx_next_column[row_ptr_L[i]];
						//   if ( i == 17051593 ) printf("sign = 1/2: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx_next_column[row_ptr_L[i]], U_diag, k);
						  xx_next_column[row_ptr_L[i]] = 0;
					  }
					 
					  pack_k = 2; 
			   }
		       
			    else
			   {
					current_column = perm_c[k];
					for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
					{
					  xx[perm_r[row_ptr[j]]] = a[j];
					}

					// if ( k == 1089581 ) printf("sign = 1: xx[1089581] = %lf\n", xx[1089581]);
					// if ( k == 1089581 ) printf("sign = 1!!!!!!!!\n");
                  
				// if ( ! assign[m] )
				// {
					row_column = seperator_in_column[k];
					// row_column = offset_U[k+1] - offset_U[k] - 1;
					//column_end = row_ptr_U[offset_U[k+1] - 2];
					column_end = row_ptr_U[offset_U[k] + row_column - 1];
				   
					for ( j = 0; j < row_column; j+=pack )
					{
					  val = j+offset_U[k];
					  column_start = row_ptr_U[val];
					//   while ( !tag[column_start] ) ;
					  temp = xx[column_start];
					  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
					  else column_number_sn = column_end-column_start+1;

					  if ( column_number_sn < thresold )
					  {
							  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
							  {
								  xx[row_ptr_L[i]] -=  temp*L[i];
							  }
						  pack = 1;
					  }
					  else
					  {
						  column_sn_end = row_ptr_U[val+column_number_sn-1];
						  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
						  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
						  column_divid = row_num % 16;
						  divid = row_num / 16;
						//   printf("fun-%d\n", column_divid);
						  row_sn_start = offset_L[column_sn_end] + column_divid + 1;

						  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
						  {
							  row_else1 = row_ptr_U[val+qqq];
							  row_else2 = row_ptr_U[val+qqq+1];
							  temp = xx[row_else1];
							  dense_vec_counter = qqq;
							  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
							  {
								  dense_vec[dense_vec_counter++] += temp*L[sss];
							  }
							  xx[row_else2] -= dense_vec[row_else2-column_start-1];
							  dense_vec[row_else2-column_start-1] = 0;
						  }
						  temp = xx[column_sn_end];
						  dense_vec_counter = column_number_sn - 1;
						  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
						  {
							  dense_vec[dense_vec_counter++] += temp*L[i];
						  }
						  dense_vec_counter = column_number_sn - 1;
						  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
						  {
							  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
							  dense_vec[dense_vec_counter] = 0;
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
								  
								  v_row.vec = _mm512_set1_pd(xx[row_else1]);
								//   v_row.vec = _mm512_set1_pd(xx[sss]);
								  v_l.vec = _mm512_load_pd(&L[row_record]);
								  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);
								//   v_l.vec = _mm512_load_pd(&L[0]);
								//   v_l_2.vec = _mm512_load_pd(&L[8]);

								  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
								  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
							  }
							   
							  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
							  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
							  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
							  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

							  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
							  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
							  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
							  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
							  
							  dividd--;
						  }
						  
						  pack = column_number_sn;
					  }    
				  
				  }
				    

					for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
					{
						U[i] = xx[row_ptr_U[i]];
						// if ( i == 11564116 ) printf("sign = 1/3: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						// if ( i == 11564061 ) printf("sign = 1: k = %d xx[%d] = %lf \n", k, row_ptr_U[i], xx[row_ptr_U[i]]);
						xx[row_ptr_U[i]] = 0;
					}

					for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					{
						L[i] = xx[row_ptr_L[i]];
						// if ( i == 17051593 ) printf("sign = 1/3: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						xx[row_ptr_L[i]] = 0;
					}  
					
					pack_k = 1;
				// }
					// tag[k] = 1;
			  }
		    
		    }
		    
		  }

	      else // parallel > serial : 3~4
		  {
	     	for ( k = start[m]; k <= end[m]; k+=pack_k )
			{
			//    k = kk;
			   if ( flag[k] )
			   {
				//    printf("sign = 2 flag != 0\n");
					//   for ( j = offset_U[k]; j < offset_U[k+1]; j++ )
					//   {
					// 	  xx[row_ptr_U[j]] = U[j];
					//   }
					//   for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
					//   {
					// 	  xx[row_ptr_L[j]] = L[j];
					//   }
					//   for ( j = offset_U[k+1]; j < offset_U[k+2]; j++ )
					//   {
					// 	  xx_next_column[row_ptr_U[j]] = U[j];
					//   }
					//   for ( j = offset_L[k+1]+1; j < offset_L[k+2]; j++ )
					//   {
					// 	  xx_next_column[row_ptr_L[j]] = L[j];
					//   }
				
					  if ( assign[m] )
					  {
						  	current_column = perm_c[k];
							current_next_column = perm_c[k+1];
							for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
							{
								xx[perm_r[row_ptr[j]]] = a[j];
							}
							for ( j = offset[current_next_column]; j < offset[current_next_column+1]; j++ )
							{
								xx_next_column[perm_r[row_ptr[j]]] = a[j];
							}
					  }

					  else
					  {
						  	for ( j = offset_U[k]; j < offset_U[k+1]; j++ )
							{
								xx[row_ptr_U[j]] = U[j];
							}
							for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
							{
								xx[row_ptr_L[j]] = L[j];
							}
							for ( j = offset_U[k+1]; j < offset_U[k+2]; j++ )
							{
								xx_next_column[row_ptr_U[j]] = U[j];
							}
							for ( j = offset_L[k+1]+1; j < offset_L[k+2]; j++ )
							{
								xx_next_column[row_ptr_L[j]] = L[j];
							}
					  }
					  
				 
					  columns_1 = seperator_in_column[k];
					  columns_next_column_1 = seperator_in_column[k+1];
					  columns = seperator_in_column_2[k];
					  column_end = row_ptr_U[offset_U[k] + columns - 1];

					  //columns_next_column_1 = seperator_in_column[k+1];
					  columns_next_column = seperator_in_column_2[k+1];
					  column_end_next_column = row_ptr_U[offset_U[k+1] + columns_next_column-1];
					  
					  for ( j = columns_1, j_next_column = columns_next_column_1; j < columns || j_next_column < columns_next_column; j+=pack, j_next_column+=pack_next_column )
					  {
						  val = j+offset_U[k];
						  val_next_column = j_next_column+offset_U[k+1];

						  column_start = row_ptr_U[val];
						  column_start_next_column = row_ptr_U[val_next_column];

						  temp = xx[column_start];
						  temp_next_column = xx_next_column[column_start_next_column];

						  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
						  else column_number_sn = column_end-column_start+1;

						  if ( column_end_next_column-column_start_next_column > sn_record[column_start_next_column] ) column_number_sn_next_column = sn_record[column_start_next_column]+1;
						  else column_number_sn_next_column = column_end_next_column-column_start_next_column+1;
						  
						  if ( column_start < column_start_next_column )
						  {
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
								  }
								  pack = 1;
								  pack_next_column = 0;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
				
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter++] += temp*L[sss];
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
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
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  dividd--;
								  }
								  pack = column_number_sn;
								  pack_next_column = 0;
							  }
						  }
						  else if ( column_start > column_start_next_column )
						  {
							  if ( column_number_sn_next_column < thresold )
							  {
								  for ( i = offset_L[column_start_next_column]+1; i < offset_L[column_start_next_column+1]; i++ )
								  {
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 0;
								  pack_next_column = 1;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val_next_column+column_number_sn_next_column-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start_next_column + 1] - offset_L[column_start_next_column] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
								  
								  for ( qqq = 0; qqq < column_number_sn_next_column-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val_next_column+qqq];
									  row_else2 = row_ptr_U[val_next_column+qqq+1];
									  temp = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec_2[dense_vec_counter++] += temp*L[sss];
									  }
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start_next_column-1];
									  dense_vec_2[row_else2-column_start_next_column-1] = 0;
								  }
								  temp = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec_2[dense_vec_counter++] += temp*L[i];
								  }
								  dense_vec_counter = column_number_sn_next_column - 1;
								  for ( sss = offset_L[column_start_next_column+1]-row_num; sss < offset_L[column_start_next_column+1]-divid*16; sss++ )
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
									  for ( sss = 0; sss < column_number_sn_next_column; sss++ )
									  {
										  row_else1 = row_ptr_U[val_next_column+sss];
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
								  
								  pack = 0;
								  pack_next_column = column_number_sn_next_column;
							  }
						  }
						  else
						  {
			
							  if ( column_number_sn < thresold )
							  {
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 1;
								  pack_next_column = 1;
							  }
							  else
							  {
								  column_sn_end = row_ptr_U[val+column_number_sn-1];
								  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
								  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
								  column_divid = row_num % 16;
								  divid = row_num / 16;
								  row_sn_start = offset_L[column_sn_end] + column_divid + 1;
								  
								  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								  {
									  row_else1 = row_ptr_U[val+qqq];
									  row_else2 = row_ptr_U[val+qqq+1];
									  temp = xx[row_else1];
									  temp_next_column = xx_next_column[row_else1];
									  dense_vec_counter = qqq;
									  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									  {
										  dense_vec[dense_vec_counter] += temp*L[sss];
										  dense_vec_2[dense_vec_counter] += temp_next_column*L[sss];
										  dense_vec_counter++;
									  }
									  xx[row_else2] -= dense_vec[row_else2-column_start-1];
									  dense_vec[row_else2-column_start-1] = 0;
									  
									  xx_next_column[row_else2] -= dense_vec_2[row_else2-column_start-1];
									  dense_vec_2[row_else2-column_start-1] = 0;
								  }
								  temp = xx[column_sn_end];
								  temp_next_column = xx_next_column[column_sn_end];
								  dense_vec_counter = column_number_sn - 1;
								  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								  {
									  dense_vec[dense_vec_counter] += temp*L[i];
									  dense_vec_2[dense_vec_counter] += temp_next_column*L[i];
									  dense_vec_counter++;
								  }
								  dense_vec_counter = column_number_sn - 1;
								  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								  {
									  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
									  dense_vec[dense_vec_counter] = 0;
									  xx_next_column[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
									  dense_vec_2[dense_vec_counter] = 0;
									  dense_vec_counter++;
								  }
								  dividd = divid;
								  for ( qqq = 0; qqq < divid; qqq++ )
								  {
									  add_sum.vec = zero.vec; 
									  add_sum_2.vec = zero.vec;
									  add_sum_3.vec = zero.vec; 
									  add_sum_4.vec = zero.vec;
									  
									  for ( sss = 0; sss < column_number_sn; sss++ )
									  {
										  row_else1 = row_ptr_U[val+sss];
										  row_record = offset_L[row_else1+1] - dividd*16;
										  
										  v_row.vec = _mm512_set1_pd(xx[row_else1]);
										  v_row_2.vec = _mm512_set1_pd(xx_next_column[row_else1]);
										  
										  v_l.vec = _mm512_load_pd(&L[row_record]);
										  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
										  
										  add_sum_3.vec = _mm512_fmadd_pd(v_row_2.vec, v_l.vec, add_sum_3.vec);	
										  add_sum_4.vec = _mm512_fmadd_pd(v_row_2.vec, v_l_2.vec, add_sum_4.vec);								  
									  }

									  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									  
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx_next_column[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum_3.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi.vec, v_sub.vec, 8);

									  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx_next_column[0], 8);
									  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_4.vec);
									  _mm512_i32scatter_pd(&xx_next_column[0], vi_2.vec, v_sub_2.vec, 8);
				 
									  dividd--;
								  }
								 
								  pack = column_number_sn;
								  pack_next_column = column_number_sn;
							  }
						  }	  
					  } 
					  
					  
					  for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
					  {
						  U[i] = xx[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 2/1: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx[row_ptr_U[i]] = 0;
					  }
					  
					  U_diag = U[i-1];
					  for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					  {
						  //L[i] = xx[row_ptr_L[i]] / U_diag;
						  L[i] = xx[row_ptr_L[i]];
						//   if ( i == 17051593 ) printf("sign = 2/1: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						  xx[row_ptr_L[i]] = 0;
					  }
					  
					  for ( i = offset_U[k+1]; i < offset_U[k+2]; i++ )
					  {
						  U[i] = xx_next_column[row_ptr_U[i]];
						//   if ( i == 11564116 ) printf("sign = 2/2: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						  xx_next_column[row_ptr_U[i]] = 0;
					  }
					  
					  U_diag_next_column = U[i-1];
					  for ( i = offset_L[k+1]+1; i < offset_L[k+2]; i++ )
					  {
						  //L[i] = xx_next_column[row_ptr_L[i]] / U_diag_next_column;
						  L[i] = xx_next_column[row_ptr_L[i]];
						//   if ( i == 17051593 ) printf("sign = 2/2: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx_next_column[row_ptr_L[i]], U_diag, k);
						  xx_next_column[row_ptr_L[i]] = 0;
					  }
					  
					  pack_k = 2;
			   }
		       
			    else
			   {
					/*for ( j = offset_U[k]; j < offset_U[k+1]; j++ )
					{
					  xx[row_ptr_U[j]] = U[j];
					}
					for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
					{
					  xx[row_ptr_L[j]] = L[j];
					}*/
					
					// if ( k == 1089582 ) printf("sign = 2/3: xx[1089586] = %lf\n", xx[1089586]);

					if ( assign[m] )
					{
							current_column = perm_c[k];
							for ( j = offset[current_column]; j < offset[current_column+1]; j++ )
							{
								xx[perm_r[row_ptr[j]]] = a[j];
							}
							// if ( k == 1089582 ) printf("sign = 2/3~~: xx[1089586] = %lf\n", xx[1089586]);
					}
					else
					{
							for ( j = offset_U[k]; j < offset_U[k+1]; j++ )
							{
							xx[row_ptr_U[j]] = U[j];
							}
							for ( j = offset_L[k]+1; j < offset_L[k+1]; j++ )
							{
							xx[row_ptr_L[j]] = L[j];
							}
					
					}
					
					// if ( k == 1089597 ) printf("sign = 2/3~: xx[1089597] = %lf\n", xx[1089597]);
					// if ( k == 1089582 ) printf("sign = 2/3~: xx[1089586] = %lf\n", xx[1089586]);
			
					// if ( k == 1089581 ) printf("sign = 2: xx[1089581] = %lf\n", xx[1089581]);

					// if ( ! assign[m] )
					// {
					row_column = seperator_in_column_2[k];
					//column_end = row_ptr_U[offset_U[k+1] - 2];
					column_end = row_ptr_U[offset_U[k] + row_column - 1];
					columns_1 = seperator_in_column[k];
					//columns_next_column_1 = seperator_in_column[k+1];
				   
					for ( j = columns_1; j < row_column; j+=pack )
					{
					  val = j+offset_U[k];
					  column_start = row_ptr_U[val];
					  temp = xx[column_start];
					  if ( column_end-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
					  else column_number_sn = column_end-column_start+1;

					  if ( column_number_sn < thresold )
					  {
							  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
							  {
								  xx[row_ptr_L[i]] -=  temp*L[i];
							  }
						  pack = 1;
					  }
					  else
					  {
						  column_sn_end = row_ptr_U[val+column_number_sn-1];
						  row_num = offset_L[column_sn_end+1] - offset_L[column_sn_end] - 1;
						  row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
						  column_divid = row_num % 16;
						  divid = row_num / 16;
						  row_sn_start = offset_L[column_sn_end] + column_divid + 1;

						  for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
						  {
							  row_else1 = row_ptr_U[val+qqq];
							  row_else2 = row_ptr_U[val+qqq+1];
							  temp = xx[row_else1];
							  dense_vec_counter = qqq;
							  for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
							  {
								  dense_vec[dense_vec_counter++] += temp*L[sss];
							  }
							  xx[row_else2] -= dense_vec[row_else2-column_start-1];
							  dense_vec[row_else2-column_start-1] = 0;
						  }
						  temp = xx[column_sn_end];
						  dense_vec_counter = column_number_sn - 1;
						  for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
						  {
							  dense_vec[dense_vec_counter++] += temp*L[i];
						  }
						  dense_vec_counter = column_number_sn - 1;
						  for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
						  {
							  xx[row_ptr_L[sss]] -= dense_vec[dense_vec_counter];
							  dense_vec[dense_vec_counter] = 0;
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
								  
								  v_row.vec = _mm512_set1_pd(xx[row_else1]);
								  v_l.vec = _mm512_load_pd(&L[row_record]);
								  v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

								  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
								  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
							  }
							   
							  vi.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
							  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
							  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
							  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

							  vi_2.vec = _mm256_load_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
							  v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
							  v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
							  _mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
							  
							  dividd--;
						  }
						  pack = column_number_sn;
					  }    
				  }
				
					for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
					{
						U[i] = xx[row_ptr_U[i]];
						// if ( i == 11564116 ) printf("sign = 2/3: xx[%d] = %lf k = %d\n", row_ptr_U[i], xx[row_ptr_U[i]], k);
						// if ( i == 11564061 ) printf("sign = 2: k = %d xx[%d] = %lf \n", k, row_ptr_U[i], xx[row_ptr_U[i]]);
						xx[row_ptr_U[i]] = 0;
					}

					U_diag = U[i-1];
					for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
					{
						// L[i] = xx[asub_L[i]] / U_diag;
						L[i] = xx[row_ptr_L[i]];
						// if ( i == 17051593 ) printf("sign = 2/3: L[i] = %lf xx[%d] = %lf U_diag = %lf k = %d\n", L[17051593], row_ptr_L[i], xx[row_ptr_L[i]], U_diag, k);
						xx[row_ptr_L[i]] = 0;
					}  
					
					pack_k = 1;
					// }
			  }
		    
		    }
		    
		  }
		   
		}
		 #pragma omp barrier
	 }
	}
	
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
/*if ( thread_number == thread_number_correct )
{
	y = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
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
  printf("fun-error of Nic results are: %d\n", error_nic); 
  for ( ij = 0; ij < nzu; ij++ )
  {
	  if ( fabs(ux[ij]-U[ij]) > 0.1 )
	  {
		  error_nic++;
		  printf("fun-nicslu[%d] = %lf me[%d] = %lf\n", ij, ux[ij], ij, U[ij]);
	  }	
  }
}*/
  /*y = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
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
		  printf("nicslu_L[%d] = %lf me_L[%d] = %lf\n", ij, lx[ij], ij, L[ij]);
	  }	
  }*/

  return U;
}

