#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>
#include <stdbool.h>
# include <math.h>

int min ( int a, int b )
{
    if ( a > b ) return b;
    return a;
}

int detect ( int *asub, int *xa, int lower_col, int higher_col )
{
    int lower_col_ptr = xa[lower_col + 1] - 1;
    int higher_col_ptr = xa[higher_col + 1] - 1;
    int lower_col_count = xa[lower_col + 1] - xa[lower_col];
    int higher_col_count = xa[higher_col + 1] - xa[higher_col];
    int count = min(lower_col_count, higher_col_count) - 1;
    int i;

    for ( i = 0; i < count; i++ )
    {
        if ( asub[lower_col_ptr] == asub[higher_col_ptr] )
        {
            lower_col_ptr--;
            higher_col_ptr--;
            continue;
        }
        else
        {
			break;
        }
    }

    if ( i == count )
    {
        if ( asub[lower_col_ptr] == higher_col && asub[higher_col_ptr] == higher_col ) return 1;
    }
    return 0;
}

void FLU_Detect_SuperNode(int *row_ptr_L, int *offset_L, int *sn_record, int *sn_number, int *sn_row_num, int n, int *sn_column_start, int *sn_column_end, int *sn_sum_finale_re, int *sum_sn_col_final)
{
    int *sn_start = (int *)malloc( sizeof( int ) * n);
    int *sn_end = (int *)malloc( sizeof( int ) * n);
	int col_thresold = 4;
    memset(sn_start, 0, sizeof(int)*n);
    memset(sn_end, 0, sizeof(int)*n);
    int sn_sum = 0;
    int sn_sum_final = 0;
    int i, j, k;
    
    int lower = 0, higher;
   
    /* detect supernode */ 
    for ( i = 0; i < n-1; i++ )
    {
		higher = i + 1;
		if ( detect(row_ptr_L, offset_L, lower, higher) )
        {
			sn_start[sn_sum] = lower;
			sn_end[sn_sum] = higher;
        } 
		else
		{
			lower = higher;
			sn_sum++;
		}
    }

    int *sn_num_record = (int *)malloc(sizeof(int) * n);
    memset(sn_num_record, -1, sizeof(int) * n); 
	int *sn_number_cou = (int *)malloc(sizeof(int) * n);
	memset(sn_number_cou, 0, sizeof(int) * n);
	double sum = 0; 
	int bad_sn = 0;
	int divid;

	int sn_div;
	col_thresold = 8;
	int thre_1 = 8; 
	// int thre_1 = 1000000000;
	int thre_2 = 8; 
	int sn_j;

	int sn_col_num, sn_num;
	int sn_s = 0;
	int j_start;

	int sum_sn_col = 0;

	for ( i = 0; i <= sn_sum; i++ )
    {
		if ( sn_end[i] - sn_start[i] >= col_thresold ) 
		{
			if ( sn_end[i] - sn_start[i] + 1 >= thre_1 )
			{
				for ( j = sn_start[i]; j <= sn_end[i]; j+=thre_2 )
				{
					if ( j+thre_2 <= sn_end[i] )
					{
						for ( sn_j = 0; sn_j < thre_2; sn_j++ )
						{
							sn_record[j+sn_j] = thre_2 - 1 - sn_j;
							sn_number[j+sn_j] = sn_sum_final;
						}
						sn_row_num[sn_sum_final] = offset_L[j+thre_2] - offset_L[j+thre_2-1] - 1;
						sum_sn_col += thre_2; 			
						sn_column_start[sn_sum_final] = j;
						sn_column_end[sn_sum_final] = j+7;
						sn_sum_final++;
						
					}
		
				}
			}
			// else
			// {
			// 	for ( j = sn_start[i]; j <= sn_end[i]; j++ )
            //     {
			// 		sn_record[j] = sn_end[i] - j;
            //         sn_number[j] = sn_sum_final;
            //         sn_number_cou[j] = sn_end[i] - sn_start[i] + 1;
            //     }
            //     sn_row_num[sn_sum_final] = offset_L[sn_end[i]+1] - offset_L[sn_end[i]] - 1;
            //     sum_sn_col += sn_end[i] - sn_start[i] + 1;
            //     sn_column_start[sn_sum_final] = sn_start[i];
            //     sn_column_end[sn_sum_final] = sn_end[i];
            //     sn_sum_final++;
			// }
				
		}
    }
    *sn_sum_finale_re = sn_sum_final;
	*sum_sn_col_final = sum_sn_col;

	// printf("sum_sn_col in Detect SN = %d\n", sum_sn_col);
}
