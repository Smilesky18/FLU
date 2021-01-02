#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
# include <math.h>

int FLU_double_computing(int *flag, char *f, int *row_ptr_U, int *offset_U, int n, int *asub_U_level, int *row_ptr_U_after_double_column)
{
    FILE *file;
    file = fopen(f, "r");
	int *flag_computation;
    flag_computation = (int *)malloc(sizeof(int) * n);
    memset(flag_computation, 0, sizeof(int) * n);
    // prior_column = (int *)malloc(sizeof(int) * n);
    // memset(prior_column, 0, sizeof(int) * n);
    int *column_counter, *column_pos_start, *column_sn_number, *column_j_number, *sn_column_start_arr, *sn_column_end_arr;
    column_counter = (int *)malloc(sizeof(int) * n);
    memset(column_counter, 0, sizeof(int) * n);
    column_pos_start = (int *)malloc(sizeof(int) * n);
    memset(column_pos_start, 0, sizeof(int) * n);
    column_sn_number = (int *)malloc(sizeof(int) * 2*n);
    memset(column_sn_number, 0, sizeof(int) * 2*n);
    column_j_number = (int *)malloc(sizeof(int) * 2*n);
    memset(column_j_number, 0, sizeof(int) * 2*n);
    sn_column_start_arr = (int *)malloc(sizeof(int) * 2*n);
    memset(sn_column_start_arr, 0, sizeof(int) * 2*n);
    sn_column_end_arr = (int *)malloc(sizeof(int) * 2*n);
    memset(sn_column_end_arr, 0, sizeof(int) * 2*n);

    int column, sn_number, j_number, columns_number, sn_column_start_file, sn_column_end_file;
    int p_file = 0;
    int sum_p_file = 0;
    int thre = 4;
	int i, j;

    while( fscanf(file, "%d %d %d %d %d %d\n", &column, &sn_number, &j_number, &columns_number, &sn_column_start_file, &sn_column_end_file) == 6 )
    {
	    if ( !column_counter[column] )
	    {
		    column_pos_start[column] = p_file;
		    column_sn_number[p_file] = sn_number;
		    column_j_number[p_file] = j_number;
		    sn_column_start_arr[p_file] = sn_column_start_file;
		    sn_column_end_arr[p_file] = sn_column_end_file;
		    column_counter[column]++;
		    p_file++;
	    }
	    else
	    {
		    column_counter[column]++;
		    column_sn_number[p_file] = sn_number;
		    column_j_number[p_file] = j_number;
		    sn_column_start_arr[p_file] = sn_column_start_file;
		    sn_column_end_arr[p_file] = sn_column_end_file;
		    p_file++;
	    }
    }

    int min_counter, start_pos, end_pos, sum_equal, pack, sum_sn_in_U, pack1, pack2, k, sum_sub, j_record, k_record, sum_sub_start;
    sum_equal = 0;
    sum_sn_in_U = 0;
    pack1 = 0;
    pack2 = 0;
	int sum_sn_in_double_col = 0;

    for ( i = 0; i < n-1; i+=pack )
    {
	    if ( !column_counter[i] & !column_counter[i+1] ) 
	    {
		    pack = 2; 
		    continue;
	    }
	    else if ( !column_counter[i] & column_counter[i+1] ) 
	    {
		    pack = 1; 
		    continue;
	    }
	    else if ( column_counter[i] & !column_counter[i+1] ) 
	    {
		    pack = 2; 
		    continue;
	    }
	    else
	    {
		    start_pos = column_pos_start[i];
		    end_pos = column_pos_start[i+1];
		    sum_equal = 0;
		    sum_sub = 0;
		    sum_sub_start = 0;
		    for ( j = start_pos, k = end_pos; j < start_pos+column_counter[i] && k < end_pos+column_counter[i+1]; j+=pack1, k+=pack2 )
		    {
			    if ( column_sn_number[j] == column_sn_number[k] )
			    {
				    pack1 = 1;
				    pack2 = 1;
				    sum_equal++;
				    if ( sn_column_start_arr[j] != sn_column_start_arr[k] ) sum_sub_start ++;
				    if ( sn_column_end_arr[j] != sn_column_end_arr[k] ) 
				    {
					    sum_sub ++;
					    j_record = j;
					    k_record = k;
				    }
			    }
			    else if ( column_sn_number[j] < column_sn_number[k] )
			    {
				    pack1 = 1;
				    pack2 = 0;
			    }
			    else
			    {
				    pack1 = 0; 
				    pack2 = 1;
			    }
		    }
		    if ( sum_equal >= thre ) 
		    {
			    if ( sum_sub > 0 || row_ptr_U[offset_U[i+2]-2] == i )
			    {
					// if ( row_ptr_U[offset_U[i+2]-2] != i ) printf(" != 0!\n");
					// printf("sum_sub = %d ", sum_sub);
				    // flag[i] = 1;
				    /*flag[i+1] = 1;
					flag_computation[i] = i;
					flag_computation[i+1] = i; */
					// if ( length[i+1] - length[i] == 1 ) 
					{
						flag[i] = 1;
						flag[i+1] = 2;
						sum_sn_in_double_col++;
					}
					// printf("%d %d\n", length[i], length[i+1]);
			    }
			    else 
			    {
				    // flag[i] = 2;
				    /*flag[i+1] = 2;
					flag_computation[i] = i;
					flag_computation[i+1] = i; */
					// if ( length[i+1] - length[i] == 1 ) 
					{
						flag[i] = 1;
						flag[i+1] = 2;
						sum_sn_in_double_col++;
					}
					// printf("%d %d\n", length[i], length[i+1]);
			    }
			    sum_sn_in_U++;
			    pack = 2; 
			    continue;
		    }
		    else 
		    {
			    pack = 1;
			    continue;
		    }
	    }
    }
	
	int n_after_double_column = n - sum_sn_in_double_col;
	// int *row_ptr_U_after_double_column = (int *)malloc(sizeof(int) * n_after_double_column);
	int row_start = 0;
	int i_pack;

	for ( i = 0; i < n; i += i_pack )
	{
		k = asub_U_level[i];
		if ( flag[k] == 2 )
		{
			i_pack = 1;
			continue;
			// row_ptr_U_after_double_column[row_start++] = k;
		}
		else
		{
			// printf("n_after = %d row_start = %d\n", n_after_double_column, row_start);
			row_ptr_U_after_double_column[row_start++] = k;
			i_pack = 1;
		}
	}
	printf("n = %d n_after_double_column = %d row_start = %d\n", n, n_after_double_column, row_start);
    printf("sum_sn_in_U = %d sum_sn_in_double_col = %d\n", sum_sn_in_U, sum_sn_in_double_col);

	return n_after_double_column;
	// printf("sn_sum_final = %d\n", sn_sum_final);
}
