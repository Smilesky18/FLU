#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
# include <math.h>

int belong( int column, int *xa_belong, int *asub_belong, int *length_belong)
{
	int i, temp = length_belong[column];

	for ( i = xa_belong[column]; i < xa_belong[column+1] - 1; i++ )
	{
		if ( length_belong[ asub_belong[i] ] > temp ) temp = length_belong[ asub_belong[i] ];
	}
	return temp+1;
}

void FLU_Dependency_Analysis(int *row_ptr_U, int *offset_U, int *asub_U_level, int *xa_trans, int prior_column_c, int num_thread, int *gp_level, int *prior_level)
{
    int *length = ( int *)malloc( sizeof(int) * prior_column_c);
	memset(length, -1, sizeof(int) * prior_column_c);
	int length_pack, len1, len2;
    int i, j;
	for ( i = 0; i < prior_column_c; i++ )
	{
		if ( offset_U[i+1] - offset_U[i] == 1 ) length[i] = 0;
		else
		{
			length[i] = belong(i, offset_U, row_ptr_U, length);
		}
	}

	int max_level = 0;
	for ( i = 0; i < prior_column_c; i++ )
	{
		if ( length[i] > max_level ) max_level = length[i];
	}
    *gp_level = max_level;

	int *level = ( int *)malloc( sizeof(int) *( max_level+1 ) );
	memset(level, 0, sizeof(int) *(  max_level+1 ));
	for ( i = 0; i < prior_column_c; i++ )
	{
		level[length[i]]++;
	}

	int *xa = ( int *)malloc( sizeof(int) * (max_level+2));
	memset(xa, 0, sizeof(int) * (max_level+2));
	for ( i = 1; i <= max_level+1; i++ )
	{
		xa[i] = xa[i-1] + level[i-1];
	}
    for ( i = 0; i <= max_level+1; i++ )
    {
        xa_trans[i] = xa[i];
    }

	for ( i = 0; i < prior_column_c; i++ )
	{
		asub_U_level[ xa[length[i]]++ ] = i;
	}

	int sum_level = max_level;
	int sum_more_than_16_columns = 0;
	for ( i = 0; i < max_level+1; i++ )
	{
		if ( level[i]  == 1 )
		{
			sum_level = i;
			break; 
		}
	}
	*prior_level = sum_level;
	printf("sum_level = %d max_level = %d\n", sum_level, max_level);
	printf("xa_trans[sum_level] = %d xa_trans[max_level] = %d\n", xa_trans[sum_level], xa_trans[max_level]);
}
