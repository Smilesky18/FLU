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

void flu_double_col_computing(double *a, int *row_ptr, int *offset, int n, int nzl, int nzu, int *perm_c, int *perm_r, int *row_ptr_L, int *offset_L, int *row_ptr_U, int *offset_U, int *sn_record, int thresold, double *L, double *U, double **xx1, double **xx2, double **dv1, double **dv2, int *asub_U_level, char *tag, int max_level, int pri_level, int *xa_trans, int num_thread, char *flag, int sum_fl, int *n_new, int *sn_number, int *sn_column_start, int *sn_column_end)
{
   omp_set_num_threads(num_thread);

  #pragma omp parallel
{
	double *xx_next_column, *xx;
	double U_diag;
    int j, k, current_column, jj;
    int i;
	int  val, columns, column_end, pack, column_start, column_number_sn, column_sn_end, row_num, row_count, column_divid, divid, row_sn_start, qqq, row_else1, row_else2, dense_vec_counter, sss, dividd, row_record, current_next_column, columns_next_column, column_end_next_column, j_next_column, pack_next_column, val_next_column, column_start_next_column, column_number_sn_next_column, pack_k, rrr, row_column, row_column_start, pack_j, column_next, row_else3, row_else4;
    double temp, temp_next_column, U_diag_next_column;
    double *dense_vec_2, *dense_vec;
	v2df_t v_l, v_row, v_mul, v_sub, v, add_sum, zero, v_sub_2, v_2, add_sum_2, v_l_2, v_row_2, add_sum_3, add_sum_4, add_sum_5, add_sum_6, add_sum_7, add_sum_8, v_l_3, v_l_4, v_3, v_4, v_sub_3, v_sub_4, x0, x1, x2, x3;
    v2if_t vi, vi_2, vi_3, vi_4;
    zero.vec = _mm512_setzero_pd();
	int kk, m, columns_1, columns_next_column_1;
	
	int tn = omp_get_thread_num();
	// int tn = 0;
	xx = xx1[tn];
	xx_next_column = xx2[tn];
	dense_vec = dv1[tn];
	dense_vec_2 = dv2[tn];
	volatile char *wait;
	volatile char *wait_2;

	int div, c_n;
	int wait_col;
	int val2;

	int sn_offset_start, sn_offset_end;

	#pragma omp for schedule(static, 1)	
	for ( kk = 0; kk < sum_fl; kk++ )
	// for ( kk = xa_trans[pri_level]; kk < xa_trans[max_level]; kk++ )
	// for ( k = 0; k < n; k++ )
	  {
		  {
			//   for ( kk = start[m]; kk <= end[m]; kk++ )
			//   for ( kk = xa_trans[m]; kk < xa_trans[m+1]; kk++ )
			  {
			  k = n_new[kk];
			//  k = asub_U_level[kk];

			  if ( !flag[k] )
			  {
				// printf("Single-column computing!\n");
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
						// printf("k = %d row_num = %d\n", k, row_num);
						row_count = offset_L[column_start + 1] - offset_L[column_start] - 1;
						column_divid = row_num % 16;
						divid = row_num / 16;
						row_sn_start = offset_L[column_sn_end] + column_divid + 1;
						// sn_offset_end = sn_column_end[sn_number[column_start]];
					//   sn_offset_start = sn_column_start[sn_number[column_start]];

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
						// for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
						{
							xx_next_column[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
							dense_vec_2[dense_vec_counter] = 0;
							dense_vec_counter++;
						}
						dividd = divid;
						// row_sn_start = offset_L[sn_offset_start] + column_divid + 8;
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
							// printf("row_sn_start = %d qqq= %d kk = %d k = %d\n", row_sn_start, qqq, kk, k);
							// printf("row_sn_start+qqq*16 = %d k = %d\n", row_sn_start+qqq*16, k);
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
				pack_k = 1;
			  }
			  
			  else 
			  {
				//   printf("Double-column computing!\n");
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
				
					  columns = offset_U[k+1] - offset_U[k] - 1;
					  column_end = row_ptr_U[offset_U[k+1] - 2];

					  columns_next_column = offset_U[k+2] - offset_U[k+1] - 2;
					  column_end_next_column = row_ptr_U[offset_U[k+2] - 3];
		  
					  for ( j = 0, j_next_column = 0; j < columns && j_next_column < columns_next_column; j+=pack, j_next_column+=pack_next_column )
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
								column_next = row_ptr_U[val];
								wait = (volatile char*)&(tag[column_next]);
								
								while ( !(*wait) ) 
								{ 
									;
								}
								  for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
								  {
									  xx[row_ptr_L[i]] -=  temp*L[i];
								  }
								  pack = 1;
								  pack_next_column = 0;
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

									  vi.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									  v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									  _mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									  vi_2.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
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
								  column_next = row_ptr_U[val_next_column];
									wait = (volatile char*)&(tag[column_next]);
									
									while ( !(*wait) ) 
									{ 
										;
									}

								  for ( i = offset_L[column_start_next_column]+1; i < offset_L[column_start_next_column+1]; i++ )
								  {
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 0;
								  pack_next_column = 1;
							  }
							  else
							  {
								   for ( jj = 0; jj < column_number_sn_next_column; jj++ )
									{
											column_next = row_ptr_U[val_next_column+jj];
											wait = (volatile char*)&(tag[column_next]);
											
											while ( !(*wait) ) 
											{ 
												;
											}
									}
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
								 
								  pack = 0;
								  pack_next_column = column_number_sn_next_column;
							  }
						  
						  }
						  
						  else
						  {
			
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
									  xx[row_ptr_L[i]] -=  temp*L[i];
									  xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
								  }
								  pack = 1;
								  pack_next_column = 1;
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
										  
										  v_l.vec = _mm512_loadu_pd(&L[row_record]);
										  v_l_2.vec = _mm512_loadu_pd(&L[row_record + 8]);

										  add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										  add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
										  
										  add_sum_3.vec = _mm512_fmadd_pd(v_row_2.vec, v_l.vec, add_sum_3.vec);	
										  add_sum_4.vec = _mm512_fmadd_pd(v_row_2.vec, v_l_2.vec, add_sum_4.vec);								  
									  }

									  vi.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									  vi_2.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									  
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
					  
					  for ( ; j < columns; j+=pack_j )
					  {
							val = j+offset_U[k];
							column_start = row_ptr_U[val];
							temp = xx[column_start];
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
										xx[row_ptr_L[i]] -=  temp*L[i];
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

								for ( qqq = 0; qqq < column_number_sn-1; qqq++ )
								{
									row_else1 = row_ptr_U[val+qqq];
									row_else2 = row_ptr_U[val+qqq+1];
									temp = xx[row_else1];
									dense_vec_counter = qqq;
									for ( sss = offset_L[row_else1]+1; sss < offset_L[row_else1+1]-divid*16; sss++ )
									{
										dense_vec_2[dense_vec_counter++] += temp*L[sss];
									}
									xx[row_else2] -= dense_vec_2[row_else2-column_start-1];
									dense_vec_2[row_else2-column_start-1] = 0;
								}
								temp = xx[column_sn_end];
								dense_vec_counter = column_number_sn - 1;
								for ( i = offset_L[column_sn_end]+1; i < offset_L[column_sn_end+1]-divid*16; i++ )
								{
									dense_vec_2[dense_vec_counter++] += temp*L[i];
								}
								dense_vec_counter = column_number_sn - 1;
								for ( sss = offset_L[column_start+1]-row_num; sss < offset_L[column_start+1]-divid*16; sss++ )
								{
									xx[row_ptr_L[sss]] -= dense_vec_2[dense_vec_counter];
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
										
										v_row.vec = _mm512_set1_pd(xx[row_else1]);
										v_l.vec = _mm512_load_pd(&L[row_record]);
										v_l_2.vec = _mm512_load_pd(&L[row_record + 8]);

										add_sum.vec = _mm512_fmadd_pd(v_row.vec, v_l.vec, add_sum.vec);	
										add_sum_2.vec = _mm512_fmadd_pd(v_row.vec, v_l_2.vec, add_sum_2.vec);	
									}
									
									vi.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16]);
									v.vec = _mm512_i32gather_pd(vi.vec, &xx[0], 8);
									v_sub.vec = _mm512_sub_pd(v.vec, add_sum.vec);
									_mm512_i32scatter_pd(&xx[0], vi.vec, v_sub.vec, 8);

									vi_2.vec = _mm256_loadu_si256((__m256i const *)&row_ptr_L[row_sn_start+qqq*16+8]);
									v_2.vec = _mm512_i32gather_pd(vi_2.vec, &xx[0], 8);
									v_sub_2.vec = _mm512_sub_pd(v_2.vec, add_sum_2.vec);
									_mm512_i32scatter_pd(&xx[0], vi_2.vec, v_sub_2.vec, 8); 
									
									dividd--;
								}
								
								pack_j = column_number_sn;
							}    
					  }
 
					  for ( ; j_next_column < columns_next_column; j_next_column+=pack_j )
					  {
							val = j_next_column+offset_U[k+1];
							column_start = row_ptr_U[val];
							temp = xx_next_column[column_start];
							if ( column_end_next_column-column_start > sn_record[column_start] ) column_number_sn = sn_record[column_start]+1;
							else column_number_sn = column_end_next_column-column_start+1;

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
					
					  if ( row_ptr_U[offset_U[k+2] - 2] != k )
					  {
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
							temp_next_column = xx_next_column[row_ptr_U[offset_U[k+2]-2]];
							column_start = row_ptr_U[offset_U[k+2] - 2];
							for ( i = offset_L[column_start]+1; i < offset_L[column_start+1]; i++ )
							{
								xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
							}
							
							for ( i = offset_U[k+1]; i < offset_U[k+2]; i++ )
							{
								U[i] = xx_next_column[row_ptr_U[i]];
								xx_next_column[row_ptr_U[i]] = 0;
							}
							
							U_diag_next_column = U[i-1];
							for ( i = offset_L[k+1]+1; i < offset_L[k+2]; i++ )
							{
								L[i] = xx_next_column[row_ptr_L[i]] / U_diag_next_column;
								xx_next_column[row_ptr_L[i]] = 0;
							}
					  }
					 
					  else
					 {
						for ( i = offset_U[k]; i < offset_U[k+1]; i++ )
						{
							U[i] = xx[row_ptr_U[i]];
							xx[row_ptr_U[i]] = 0;
						}
						
						// temp_next_column = xx_next_column[row_ptr_U[offset_U[k+2]-2]];
						  temp_next_column = xx_next_column[k];
						U_diag = U[i-1];
						for ( i = offset_L[k]+1; i < offset_L[k+1]; i++ )
						{
							L[i] = xx[row_ptr_L[i]] / U_diag;
							xx[row_ptr_L[i]] = 0;
							xx_next_column[row_ptr_L[i]] -=  temp_next_column*L[i];
						}
						
						for ( i = offset_U[k+1]; i < offset_U[k+2]; i++ )
						{
							U[i] = xx_next_column[row_ptr_U[i]];
							xx_next_column[row_ptr_U[i]] = 0;
						}
						
						U_diag_next_column = U[i-1];
						for ( i = offset_L[k+1]+1; i < offset_L[k+2]; i++ )
						{
							L[i] = xx_next_column[row_ptr_L[i]] / U_diag_next_column;
							xx_next_column[row_ptr_L[i]] = 0;
						} 
					 }
					 
				      tag[k] = 1;
					  tag[k+1] = 1;

					  pack_k = 2;
			  }
		  
		  }
		    
		  }
		  
	 }
}
}

