/*this demo is the same as demo.cpp except for that this demo calls the C interface*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>
# include <math.h>
#define MICRO_IN_SEC 1000000.00

// typedef struct {
// 	unsigned int bit32;
// }

/*unsigned int setBit(unsigned int a, int b)
{
	unsigned int tmp = 1 << b;
	return tmp 

	
}*/

typedef union{
	unsigned int bit32;
	char boolvec[4];
} bitInt;


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

int belong( int column, int *xa_belong, int *asub_belong, int *length_belong)
{
	int i, temp = length_belong[column];

	for ( i = xa_belong[column]; i < xa_belong[column+1] - 1; i++ )
	{
		if ( length_belong[ asub_belong[i] ] > temp ) temp = length_belong[ asub_belong[i] ];
	}
	return temp+1;
}

double Abs(double x)
{
	  return x < 0 ? -x : x;
} 
bool equal( double a, double b )
{
	  if ( Abs(a-b) < 0.1 )
	  {
		  return true;
	  }
	  else
	  {
		  return false;
	  }
}
int min ( int a, int b )
{
    if ( a > b ) return b;
    return a;
}
int max ( int a, int b )
{
    if ( a > b ) return a;
    return b;
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
int detect_U ( int *asub, int *xa, int lower_col, int higher_col, int *flag )
{
    if ( !flag[lower_col] || !flag[higher_col] ) return 0;
    int lower_col_ptr = xa[lower_col];
    int higher_col_ptr = xa[higher_col];
    int lower_col_count = xa[lower_col + 1] - xa[lower_col];
    int higher_col_count = xa[higher_col + 1] - xa[higher_col];
    int count = min(lower_col_count, higher_col_count);
    int i;
    int sum = 0;

    for ( i = 0; i < count; i++ )
    {
        if ( asub[lower_col_ptr] == asub[higher_col_ptr] )
        {
            lower_col_ptr++;
            higher_col_ptr++;
	    sum++;
        }
	else if ( asub[lower_col_ptr] < asub[higher_col_ptr] )
	{
		lower_col_ptr++;
	}
        else
        {
		higher_col_ptr++;
        }
    }

    if ( count - sum <= 1 ) return 1;
    return 0;
}
 

/* Time Stamp */
double microtime()
{
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

const char *const ORDERING_METHODS[] = { "AMD", "mAMD", "AMF","mAMF1","mAMF2","mAMF3" };

int main( int argc[], char *argv[])
{
    _double_t *ax = NULL, *b = NULL, *x = NULL, *x_lu_gp = NULL;
    _uint_t *ai = NULL, *ap = NULL;
    _uint_t n, row, col, nz, nnz, i, j;
    _handle_t solver = NULL;
    _double_t res[4], cond, det1, det2, fflop, sflop;
    size_t mem;
    _double_t *cfg;
    const _double_t *stat;
    const char *last_err;
    _uint_t *row_perm, *row_perm_inv, *col_perm, *col_perm_inv, *ui, *li;
    _double_t *lx, *ux, *row_scale, *col_scale;
    _size_t *lp, *up;
    _bool_t sort = 1;

    /*read matrix A*/
    if (__FAIL(ReadMatrixMarketFile(argv[1], &row, &col, &nz, NULL, NULL, NULL, NULL, NULL, NULL)))
    {
        printf("Failed to read matrix A\n");
        goto EXIT;
    }
    n = row;
    nnz = nz;
    ax = (_double_t *)malloc(sizeof(_double_t)*nnz);
    ai = (_uint_t *)malloc(sizeof(_uint_t)*nnz);
    ap = (_uint_t *)malloc(sizeof(_uint_t)*(1 + n));
    
    row_perm = (_uint_t *)malloc(sizeof(_uint_t)*n);
    col_perm = (_uint_t *)malloc(sizeof(_uint_t)*n);
    col_perm_inv = (_uint_t *)malloc(sizeof(_uint_t)*n);
    row_perm_inv = (_uint_t *)malloc(sizeof(_uint_t)*n);
    row_scale = (_double_t *)malloc(sizeof(_double_t)*n);
    col_scale = (_double_t *)malloc(sizeof(_double_t)*n); 
    lp = (_size_t *)malloc(sizeof(_size_t)*(1+n));
    up = (_size_t *)malloc(sizeof(_size_t)*(1+n));
	
    ReadMatrixMarketFile(argv[1], &row, &col, &nz, ax, ai, ap, NULL, NULL, NULL);
    printf("***********%s: row %d, col %d, nnz %d\n", argv[1], n, n, nnz);

    /*read RHS B*/
    b = (_double_t *)malloc(sizeof(_double_t)*n);
    for ( i = 0; i < n; i++ ) b[i] = 1.0;
    x = (_double_t *)malloc(sizeof(_double_t)*n);
    memset(x, 0, sizeof(_double_t) * n);
    x_lu_gp = (_double_t *)malloc(sizeof(_double_t)*n);
    memset(x_lu_gp, 0, sizeof(_double_t) * n);

    /*initialize solver*/
    if (__FAIL(NicsLU_Initialize(&solver, &cfg, &stat, &last_err)))
    {
        printf("Failed to initialize\n");
        goto EXIT;
    }
    cfg[0] = 1.; /*enable timer*/

    /*pre-ordering (do only once)*/
    NicsLU_Analyze(solver, n, ax, ai, ap, MATRIX_COLUMN_REAL, NULL, NULL, NULL, NULL);

    /*create threads (do only once)*/
    NicsLU_CreateThreads(solver, 0); /*use all physical cores*/
    /*factor & solve (first-time)*/
    NicsLU_FactorizeMatrix(solver, ax, 0); /*use all created threads*/

    /* Get L/U structural information from NicSLU */
    lx = (_double_t *)malloc(sizeof(_double_t) * stat[9]);
    ux = (_double_t *)malloc(sizeof(_double_t) * stat[10]);
    li = (_uint_t *)malloc(sizeof(_uint_t) * stat[9]);
    ui = (_uint_t *)malloc(sizeof(_uint_t) * stat[10]);
    NicsLU_GetFactors(solver, lx, li, lp, ux, ui, up, sort, row_perm, col_perm, row_scale, col_scale);
    for (i = 0; i < n; i++) col_perm_inv[col_perm[i]] = i;
    for (i = 0; i < n; i++) row_perm_inv[row_perm[i]] = i;

	printf("1111111111111111111111111\n");
    
    int lnz = (int)stat[10]; // Number of non-zeros in L 
    int unz = (int)stat[9];  // Numbe of non-zeros in U
    int *offset_U = (int *)malloc(sizeof(int) * (n+1)); // This array stores the column offset in U 
    int *offset_L = (int *)malloc(sizeof(int) * (n+1)); // This array stores the column offset in L
    for ( i = 0; i <= n; i++ ) offset_U[i] = lp[i];	// assignment: get lp from NicSLU 
    for ( i = 0; i <= n; i++ ) offset_L[i] = up[i];	// assignment: get up from NicSLU

    int *row_ptr_L = (int *)malloc(sizeof(int) * lnz);	// This array stores the row ptr in L
    int *row_ptr_U = (int *)malloc(sizeof(int) * unz);	// This array stores the row ptr in U
    for ( i = 0; i < lnz; i++ ) row_ptr_L[i] = ui[i];	// assignment: get ui from NicSLU
    for ( i = 0; i < unz; i++ ) row_ptr_U[i] = li[i];	// assignment: get li from NicSLU
    
    int *sn_record = (int *)malloc(sizeof(int) * n);	
    memset(sn_record, 0, sizeof(int) * n);
    int *sn_start = (int *)malloc( sizeof( int ) * n);
    int *sn_end = (int *)malloc( sizeof( int ) * n);
    int col_thresold = atoi(argv[2]);
    memset(sn_start, 0, sizeof(int)*n);
    memset(sn_end, 0, sizeof(int)*n);
    int sn_sum = 0;
    int sn_sum_final = 0;
    
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

    int *sn_column_start = (int *)malloc(sizeof(int) * sn_sum);
    memset(sn_column_start, 0, sizeof(int) * sn_sum);
    int *sn_column_end = (int *)malloc(sizeof(int) * sn_sum);
    memset(sn_column_end, 0, sizeof(int) * sn_sum);
    int *sn_num_record = (int *)malloc(sizeof(int) * n);
    memset(sn_num_record, -1, sizeof(int) * n); 

    double sum = 0; 
    for ( i = 0; i <= sn_sum; i++ )
    {
	if ( sn_end[i] - sn_start[i] >= col_thresold ) 
	{
		for ( j = sn_start[i]; j <= sn_end[i]; j++ )
            	{
                	sn_record[j] = sn_end[i] - j;
					sn_num_record[j] = sn_sum_final;
            	}
		sn_column_start[sn_sum_final] = sn_start[i];
		sn_column_end[sn_sum_final] = sn_end[i];
        sn_sum_final++;
		sum += sn_end[i] - sn_start[i] + 1;
	}
    }

	printf("sn_sum_final = %d\n", sn_sum_final);
     
  /*	for ( i = 0; i < sn_sum_final; i++ )
	{
		printf("supernode %d: %d~%d\n", i, sn_column_start[i], sn_column_end[i]);
	} */

    /* file process */
    FILE *file;
    file = fopen(argv[4], "r");
    int *flag;
    flag = (int *)malloc(sizeof(int) * n);
    memset(flag, 0, sizeof(int) * n);
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

	printf("33333333333333333333333333333333\n");

    int column, sn_number, j_number, columns_number, sn_column_start_file, sn_column_end_file;
    int p_file = 0;
    int sum_p_file = 0;
    int thre = atoi(argv[5]);

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
			// printf("p_file = %d sn_column_end_file = %d\n", p_file, sn_column_end_file);
		    sn_column_end_arr[p_file] = sn_column_end_file;
		    p_file++;
	    }
    }

	printf("444444444444444444444444444444444444\n");

    int min_counter, start_pos, end_pos, sum_equal, pack, sum_sn_in_U, pack1, pack2, k, sum_sub, j_record, k_record, sum_sub_start;
    sum_equal = 0;
    sum_sn_in_U = 0;
    pack1 = 0;
    pack2 = 0;

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
					if ( row_ptr_U[offset_U[i+2]-2] != i ) printf(" != 0!\n");
					// printf("sum_sub = %d ", sum_sub);
				    flag[i] = 1;
				    /*flag[i+1] = 1;
					flag_computation[i] = i;
					flag_computation[i+1] = i; */
					// printf("i = %d i+1= %d\n", i, i+1);
			    }
			    else 
			    {
				    flag[i] = 2;
				    /*flag[i+1] = 2;
					flag_computation[i] = i;
					flag_computation[i+1] = i; */
					// printf("i = %d i+1= %d\n", i, i+1);
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
	
    printf("sum_sn_in_U = %d\n", sum_sn_in_U);
	
	int l, m;
	int prior_column_c = atoi(argv[7]);
	int *length = ( int *)malloc( sizeof(int) * n);
	memset(length, -1, sizeof(int) * n);
	int length_pack, len1, len2;
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

	int *level = ( int *)malloc( sizeof(int) *( max_level+1 ) );
	memset(level, 0, sizeof(int) *(  max_level+1 ));
	for ( i = 0; i < prior_column_c; i++ )
	{
		level[length[i]]++;
	}
	printf("max_level = %d\n", max_level);
	
	// int sum_level = 0;
	// int sum_level_index;
	// for ( i = 0; i < max_level+1; i++ )
	// {
	// 	printf("level[%d] = %d\n", i, level[i]);
	// 	// if ( level[i] == 1 )
	// 	// {
	// 	// 	sum_level += level[i];
	// 	// 	sum_level_index = i;
	// 	// 	//printf("%d\n", i);
	// 	// 	break;
	// 	// }
	// }
	// printf("sum_level = %d sum_level_index = %d max_level = %d\n", sum_level, sum_level_index, max_level);

	int *xa = ( int *)malloc( sizeof(int) * (max_level+2));
	memset(xa, 0, sizeof(int) * (max_level+2));
	 
	for ( i = 1; i <= max_level+1; i++ )
	{
		xa[i] = xa[i-1] + level[i-1];
	}
	printf("xa[max_level+1] = %d\n", xa[max_level+1]);
	printf("xa[max_level+2] = %d\n", xa[max_level+2]);
	int *xa_trans = (int *)malloc( sizeof(int) * (max_level+2));
	for ( i = 0; i < max_level+2; i++ ) xa_trans[i] = xa[i];
	int *asub_U_level = (int *)malloc(sizeof(int) * prior_column_c);

	for ( i = 0; i < prior_column_c; i++ )
	{
		asub_U_level[ xa[length[i]]++ ] = i;
	}
	int num_thread = atoi(argv[9]);
	int sum_level = max_level;
	int sum_more_than_16_columns = 0;
	for ( i = 0; i < max_level+1; i++ )
	{
		if ( level[i] < num_thread )
		// if ( level[i] == 1 )
		{
			// sum_level += level[i];
			// sum_more_than_16_columns += level[i];
			sum_level = i;
			break; 
			// printf("level[%d] = %d\n", i, level[i]);
		}
		else
		{
			sum_more_than_16_columns += level[i];
		}
	}
	printf("parallel part has %d columns paralle+wait has %d columns!\n", sum_more_than_16_columns, prior_column_c);
	printf("max_level = %d sum_level = %d\n", max_level, sum_level);
	// printf("From level %d, columns in every level are less than 16\n", sum_level);
	// printf("There are %d columns in all levels which are more than 16\n", sum_more_than_16_columns);
	printf("xa_trans[%d] = %d\n", sum_level, xa_trans[sum_level]);

	char *no_wait = (char *)malloc(sizeof(char) * n);
	memset(no_wait, 0, sizeof(char) * n);
	int sum_no_wait = 0;
	int sum_whe_wait = 0;
	int j1, j2, j3, j4;

	for ( i = 0; i < xa_trans[sum_level+1]; i++ )
	{
		no_wait[asub_U_level[i]] = 1;
	}
	for ( i = xa_trans[sum_level+1]; i < xa_trans[max_level+1]; i+=num_thread )
	{
		if ( i+num_thread < xa_trans[max_level+1] )
		{
		for ( j1 = i; j1 < i+num_thread; j1++ )
		{
			j2 = asub_U_level[j1];
			// printf("j2 = %d\n", j2);
			for ( l = offset_U[j2]; l < offset_U[j2+1]; l++ )
			{
				if ( !no_wait[row_ptr_U[l]] )
				{
					sum_no_wait++;
				}
				sum_whe_wait++;
			}
		}
		for ( j1 = i; j1 < i+num_thread; j1++ )
		{
			no_wait[asub_U_level[j1]] = 1;
		}
		}
		// printf("asub_U_level[%d] = %d\n", i, asub_U_level[i]);
	}
	printf("sum_wait = %d all = %d\n", sum_no_wait, sum_whe_wait);
	/*for ( i = max_level - 5; i < max_level+1; i++ )
	{
		for ( j = xa_trans[i]; j < xa_trans[i+1]; j++ )
		{
			k = asub_U_level[j];
			//printf("%d %d\n", asub_U_level[j], flag[asub_U_level[j]]);
		}
		printf("**********\n");
	}*/

	double sum_1_number = 0;
	double sum_2_number = 0;
	double sum_3_number = 0;
	double sum_4_number = 0;
	double sum_5_number = 0;
	double sum_number = 0;
	int sum_number_counter = 0;

	double sum_compu = 1;
	int *flag_2 = (int *)malloc(sizeof(int) * n);
	memset(flag_2, 0, sizeof(int) * n);
	int *depend_column_start, *depend_column_end;
	int depend_column_number = 0;
	depend_column_start = (int *)malloc(sizeof(int) * n);
	depend_column_end = (int *)malloc(sizeof(int) * n);
	memset(depend_column_end, -1, sizeof(int) * n);
	depend_column_start[0] = 0;
	//int prior_column_c = atoi(argv[7]);
	int next_column = atoi(argv[8]);
	// int thread_divid = atoi(argv[9]);

	double thresold_2 = atof(argv[10]);
	int sum_flag = 0;
	int i_counter = 0;
	
	int prior_column = prior_column_c;
	
	/*for ( i = prior_column; i < n; i++ )
	{
		for ( j = offset_U[i]; j < offset_U[i+1] - 1; j++ )
		{
			k = row_ptr_U[j];
			sum_1_number += offset_L[k+1] - offset_L[k] - 1;
		}
		printf("column %d has %lf computations\n", i, sum_1_number);
		sum_1_number = 0;
	}
    */

	// int intmer = prior_column;

    int ii, iii, jj, kk, kkk;
	sum_number = 0;
	int thread_number = 0;
	int pack_i;
	int *start = (int *)malloc(sizeof(int) * n);
	int *end = (int *)malloc(sizeof(int) * n);
	int *seperator_in_column = (int *)malloc(sizeof(int) * n);
	memset(seperator_in_column, 0, sizeof(int) * n);
	int *seperator_in_column_2 = (int *)malloc(sizeof(int) * n);
	memset(seperator_in_column_2, 0, sizeof(int) * n);
	start[0] = 0;
	int sum_u_nnz = 0;
	int sum_square_computation_counter = 0;
	int start_i = prior_column;
	double *sum_square_computation = (double *)malloc(sizeof(double) * n);
	double *sum_square_computation_1 = (double *)malloc(sizeof(double) * n);
	double *sum_square_computation_2 = (double *)malloc(sizeof(double) * n);
	double *sum_tri_computation = (double *)malloc(sizeof(double) * n);
	int *sum_square_computation_start_index = (int *)malloc(sizeof(int) * n);
	int *sum_square_computation_end_index = (int *)malloc(sizeof(int) * n);
	double *sum_prior_level = (double *)malloc(sizeof(double) * (max_level+1));
	int prior_column_n;
	int thread_number_prior = 0;
	int thread_number_cha, thr_i;
	int prior_column_prior_level;

	for ( kk = 0; kk < sum_level+1; kk++ )
	{
		for ( kkk = xa_trans[kk]; kkk < xa_trans[kk+1]; kkk++ )
		{
			i = asub_U_level[kkk];
			for ( j = offset_U[i]; j < offset_U[i+1] - 1; j++ )
			{
				k = row_ptr_U[j];
				sum_2_number += offset_L[k+1] - offset_L[k] - 1;
			}
			sum_2_number += offset_L[i+1] - offset_L[i] - 1;
		}
		sum_prior_level[kk] = sum_2_number;
		sum_2_number = 0;
	}
	sum_2_number = 0;
	for ( kk = 0; kk < sum_level+1; kk++ )
	{
		start[thread_number] = xa_trans[kk]; 
		for ( kkk = xa_trans[kk]; kkk < xa_trans[kk+1]; kkk++ )
		{
			i = asub_U_level[kkk];
			for ( j = offset_U[i]; j < offset_U[i+1] - 1; j++ )
			{
				k = row_ptr_U[j];
				sum_2_number += offset_L[k+1] - offset_L[k] - 1;
			}
			sum_2_number += offset_L[i+1] - offset_L[i] - 1;
			if ( sum_2_number >= sum_prior_level[kk]/num_thread || kkk == xa_trans[kk+1] - 1)
			{
				// printf("computation = %lf thread_number = %d\n", sum_2_number, thread_number);
				end[thread_number] = kkk;
				thread_number++;
				start[thread_number] = kkk+1;
				sum_2_number = 0;
			}
		}
		//printf("prior = %d current = %d\n", thread_number_prior, thread_number);
		if ( thread_number-thread_number_prior != num_thread )
		{
			thread_number_cha = num_thread - thread_number + thread_number_prior;
			for ( thr_i = thread_number; thr_i < thread_number+thread_number_cha; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
			}
			thread_number += thread_number_cha;
			//printf("%d %d %d\n", thread_number, thread_number_prior, thread_number-thread_number_prior);
		}
		/*for ( thr_i = thread_number; thr_i < thread_number+thread_divid; thr_i++ )
		{
				start[thr_i] = 0;
				end[thr_i] = -1;
		}
		thread_number += thread_divid;	*/

		thread_number_prior = thread_number; 
	}
	/*start[thread_number] = xa_trans[sum_level_index];
	end[thread_number] = xa_trans[max_level+1] - 1;
	for ( i = thread_number+1; i < thread_number+thread_divid; i++ )
	{
		start[i] = 0;
		end[i] = -1;
	}
	thread_number += thread_divid;*/
	/*for ( i = 0; i < thread_number; i++ )
	{
		printf("start[%d] = %d end[%d] = %d\n", i, start[i], i, end[i]);
	}*/
	//printf("xa_trans[sum_level_index] = %d\n", xa_trans[sum_level_index]);
	printf("xa_trans[max_level+1] - 1 = %d\n", xa_trans[max_level+1] - 1);
	printf("start[0] = %d end[0] = %d\n", start[0], end[0]);

	//seperator_in_column[prior_column] = offset_U[prior_column+1] - offset_U[prior_column] - 1;

	printf("prior_column = %d prior_column_c = %d next_column = %d\n", prior_column, prior_column_c, next_column);
	sum_square_computation_start_index[0] = prior_column;
	int step = atoi(argv[11]);
	for ( i = prior_column; i < next_column; i+=step )
	{
		if ( i+step > n ) iii = n;
		//else if ( flag_computation[i+31] == flag_computation[i+32] && flag_computation[i+31] )
		else if ( flag[i+step-1] )
		{
			iii = i+step+1;
		}
		else iii = i+step;
		for ( ii = i; ii < iii; ii++ )
		{
			for ( j = offset_U[ii]; j < offset_U[ii+1] - 1; j++ )
			{
				k = row_ptr_U[j];
				if ( k <= prior_column-1 )
				{
					sum_4_number += offset_L[k+1] - offset_L[k] - 1;
				}
				sum_5_number += offset_L[k+1] - offset_L[k] - 1;
			}
			//sum_5_number += offset_L[ii+1] - offset_L[ii] - 1;
		}
	
		if ( sum_4_number / sum_5_number <= thresold_2 && sum_4_number > 0 || ii == n )
		{
			prior_column = ii;
			// printf("start = %d end = %d square_computations = %lf tri_computations = %lf square/all = %lf%%\n", i, ii, sum_4_number, sum_5_number-sum_4_number, sum_4_number/sum_5_number*100);
			sum_square_computation[sum_square_computation_counter] = sum_4_number;
			sum_tri_computation[sum_square_computation_counter] = sum_5_number - sum_4_number;
			sum_square_computation_end_index[sum_square_computation_counter] = ii;
			sum_square_computation_counter++;
			sum_square_computation_start_index[sum_square_computation_counter] = ii;
			
			sum_4_number = 0;
			sum_5_number = 0;
		}
	}
	printf("sum_square_computation_start_index[%d] = %d sum_square_computation_end_index[%d] = %d\n", sum_square_computation_counter-1, sum_square_computation_start_index[sum_square_computation_counter-1],sum_square_computation_counter-1, sum_square_computation_end_index[sum_square_computation_counter-1]);
	sum_4_number = 0;
	double sum_tri = 0;
	for ( i = 0; i < sum_square_computation_counter; i++ )
	{
		sum_tri += sum_tri_computation[i];
	}
	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!sum_tri = %lf\n", sum_tri);
    for ( i = 1; i < sum_square_computation_counter; i++ )
	{
		int s_index = sum_square_computation_start_index[i];
		int e_index = sum_square_computation_end_index[i];
		prior_column_prior_level = sum_square_computation_start_index[i-1];
		//prior_column = sum_square_computation_start_index[i];
		for ( j = s_index; j < e_index; j++ )
		{
			for ( k = offset_U[j]; k < offset_U[j+1]; k++ )
			{
				ii = row_ptr_U[k];
				if ( ii <= prior_column_prior_level-1 )
				{
					sum_4_number += offset_L[ii+1] - offset_L[ii] - 1;
				}
				else break;
			}
		}
		sum_square_computation_1[i] = sum_4_number;
		sum_square_computation_2[i] = sum_square_computation[i] - sum_4_number;
		sum_4_number = 0;
	} 
	sum_square_computation_1[0] = 0;
	sum_square_computation_2[0] = sum_square_computation[0];

	/*for ( i = 0; i < sum_square_computation_counter-1; i++ )
	{
		printf("squ1: %lf - squ2: %lf\n", sum_square_computation_1[i], sum_square_computation_2[i]);
	}*/
	
	thread_number_prior = thread_number;
	int thread_number_prior_level = thread_number;
	int thread_number_counter = 0;
	sum_number = 0;
	//int *sign_squ_1 = (int *)malloc(sizeof(int) * n);
	//int *sign_squ_2 = (int *)malloc(sizeof(int) * n);
	int *sign = (int *)malloc(sizeof(int) * n);
	memset(sign, 0, sizeof(int) * n);
	printf("thread_number = %d\n", thread_number);

	int *assign = (int *)malloc(sizeof(int) * n);
	memset(assign, 0, sizeof(int) * n);
    
	/*去掉 flag ！= 0 的列*/
	/*double divid_temp, divid_temp_ratio;
	for ( jj = 0; jj < 1; jj++ )
	{
		prior_column = sum_square_computation_start_index[jj];
		prior_column_n = sum_square_computation_end_index[jj];
		sum_4_number = sum_square_computation[jj];
		start[thread_number] = prior_column;
		divid_temp = sum_4_number/thread_divid;
		divid_temp_ratio = divid_temp;
		printf("First square computation = %lf\n", sum_4_number);
	    
		if ( sum_4_number > 0 )
		{
		for ( i = prior_column; i < prior_column_n; i+=pack_i )
		{
			// else
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i] = j - offset_U[i];
						seperator_in_column_2[i] = j - offset_U[i];
						break;
					}
				}
				if ( sum_number >= divid_temp_ratio || i == prior_column_n-1 )
				{
					end[thread_number] = i;
					sign[thread_number] = 1;
					//sign[thread_number] = 2;
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+1;
					printf("thr_num = %d computaion = %lf\n", thread_number, sum_number);
					sum_number = 0;
				}
				pack_i = 1;
			}		 
		}
		}
	
	    else
		{
			for ( thr_i = thread_number; thr_i < thread_number+num_thread; thr_i++ )
			{
				start[thr_i] = 0;
				sign[thr_i] = 1;
				end[thr_i] = -1;
			}
			thread_number += num_thread;
			thread_number_counter += num_thread;
		}
		
		if ( thread_number_counter != thread_divid )
		{
			thread_number_cha = thread_divid - thread_number_counter;
			for ( thr_i = thread_number; thr_i < thread_number+thread_number_cha; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
				sign[thr_i] = 1;
			}
			thread_number += thread_number_cha;
		}
		start[thread_number] = prior_column;
		end[thread_number] = prior_column_n-1;
		sign[thread_number] = 0;
		thread_number++;
		//printf("$$$$%d %d start[%d] = %d end[%d] = %d\n", prior_column, prior_column_n-1, thread_number, start[thread_number], thread_number, end[thread_number]);
	}
    */
    
	/*原始计算，也就是加上 flag ！= 0 的列*/
	for ( jj = 0; jj < 1; jj++ )
	{
		prior_column = sum_square_computation_start_index[jj];
		prior_column_n = sum_square_computation_end_index[jj];
		sum_4_number = sum_square_computation[jj];
		start[thread_number] = prior_column;
	    
		if ( sum_4_number > 0 )
		{
		for ( i = prior_column; i < prior_column_n; i+=pack_i )
		{
			//seperator_in_column[i] = 0;
			if ( flag[i] )
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i] = j - offset_U[i];
						seperator_in_column_2[i] = j - offset_U[i];
						break;
					}
				}
				for ( j = offset_U[i+1]; j < offset_U[i+2]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i+1] = j - offset_U[i+1];
						seperator_in_column_2[i+1] = j - offset_U[i+1];
						break;
					}
				}
				if ( sum_number >= sum_4_number/num_thread|| i+1 == prior_column_n-1 )
				{
					end[thread_number] = i+1;
					sign[thread_number] = 1;
					// printf("thr_num = %d thr_computaion = %lf all_computation = %lf tri_computation = %lf\n", thread_number, sum_number, sum_4_number, sum_tri_computation[jj]);
					//sign[thread_number] = 2;
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+2;
					// printf("thr_num = %d computaion = %lf\n", thread_number, sum_number);
					sum_number = 0;
				}
				pack_i = 2;
			}
			else
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i] = j - offset_U[i];
						seperator_in_column_2[i] = j - offset_U[i];
						break;
					}
				}
				if ( sum_number >= sum_4_number/num_thread || i == prior_column_n-1 )
				{
					end[thread_number] = i;
					sign[thread_number] = 1;
					//sign[thread_number] = 2;
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+1;
					// printf("thr_num = %d computaion = %lf\n", thread_number, sum_number);
					sum_number = 0;
				}
				pack_i = 1;
			}		 
		}
		}
	
	    else
		{
			for ( thr_i = thread_number; thr_i < thread_number+num_thread; thr_i++ )
			{
				start[thr_i] = 0;
				sign[thr_i] = 1;
				end[thr_i] = -1;
			}
			thread_number += num_thread;
			thread_number_counter += num_thread;
		}
		
		if ( thread_number_counter != num_thread )
		{
			thread_number_cha = num_thread - thread_number_counter;
			for ( thr_i = thread_number; thr_i < thread_number+thread_number_cha; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
				sign[thr_i] = 1;
			}
			thread_number += thread_number_cha;
		}
		start[thread_number] = prior_column;
		end[thread_number] = prior_column_n-1;
		sign[thread_number] = 0;
		thread_number++;
		//printf("$$$$%d %d start[%d] = %d end[%d] = %d\n", prior_column, prior_column_n-1, thread_number, start[thread_number], thread_number, end[thread_number]);
	}
	
/*	int level_var = atoi(argv[12]);
	int level_var_end = atoi(argv[13]);
	*/
	for ( jj = 1; jj < sum_square_computation_counter; jj++ )
	{
		prior_column = sum_square_computation_start_index[jj];
		prior_column_prior_level = sum_square_computation_start_index[jj-1];
		prior_column_n = sum_square_computation_end_index[jj];
		sum_4_number = sum_square_computation_1[jj];
		sum_3_number = sum_tri_computation[jj-1];
		sum_5_number = sum_square_computation_2[jj];
		start[thread_number] = prior_column;
		thread_number_counter = 0;
		sum_number = 0;
		// if ( prior_column == 3401536 ) printf("sum_4 = %lf sum_5 = %lf\n", sum_4_number, sum_5_number);
		// printf("All square computations = %lf\n", sum_4_number);
	    
		if ( sum_4_number > 0 )
		{
		for ( i = prior_column; i < prior_column_n; i+=pack_i )
		{
			if ( flag[i] )
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column_prior_level-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i] = j - offset_U[i];
						break;
					}
				}
				for ( j = offset_U[i+1]; j < offset_U[i+2]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column_prior_level-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i+1] = j - offset_U[i+1];
						break;
					}
				}
				if ( sum_number >= sum_4_number/(num_thread-1) || i+1 == prior_column_n-1 )
				{
					end[thread_number] = i+1;
					sign[thread_number] = 1;
					// if ( thread_number >= level_var && thread_number <= level_var_end ) printf("thr_num = %d computaion = %lf sum_square_computation = %lf sum_tri_computation_prior = %lf\n", thread_number, sum_number, sum_4_number, sum_3_number);
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+2;
					sum_number = 0;
				}
				pack_i = 2;
			}
			else
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column_prior_level-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else
					{
						seperator_in_column[i] = j - offset_U[i];
						break;
					}
				}
				if ( sum_number >= sum_4_number/(num_thread-1) || i == prior_column_n-1 )
				{
					end[thread_number] = i;
					sign[thread_number] = 1;
					// if ( thread_number >= level_var && thread_number <= level_var_end ) printf("thr_num = %d computaion = %lf sum_square_computation = %lf sum_tri_computation_prior = %lf\n", thread_number, sum_number, sum_4_number, sum_3_number);
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+1;
					sum_number = 0;
				}
				pack_i = 1;
			}		 
		}
		}
	
	    else
		{
			/*int temp_val = (prior_column_n-prior_column)/num_thread;
			for ( i = prior_column+temp_val; i < prior_column_n; i+=temp_val )
			{
				end[thread_number] = i;
				thread_number++;
				start[thread_number] = i+1;
			}
			end[thread_number-1] = prior_column_n-1;*/

			/*	start[thread_number] = prior_column;
				end[thread_number] = prior_column_n - 1;
				assign[thread_number] = 1;
				sign[thread_number] = 1;

			for ( thr_i = thread_number+1; thr_i < thread_number+num_thread-1; thr_i++ )*/
			for ( thr_i = thread_number; thr_i < thread_number+num_thread-1; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
				sign[thr_i] = 1;
			}
			thread_number += num_thread-1;
			thread_number_counter += num_thread-1;
		}
		
		//printf("thread_number_counter = %d\n", thread_number_counter);
		if ( thread_number_counter != num_thread - 1 )
		{
			thread_number_cha = num_thread - 1 - thread_number_counter;
			for ( thr_i = thread_number; thr_i < thread_number+thread_number_cha; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
				sign[thr_i] = 1;
			}
			thread_number += thread_number_cha;
		}
		//printf("1-thread_number_counter = %d thread_number = %d\n", thread_number_counter, thread_number);
		/*start[thread_number] = prior_column;
		end[thread_number] = prior_column_n-1;
        for ( thr_i = thread_number+1; thr_i < thread_number+thread_divid; thr_i++ )
		{
				start[thr_i] = 0;
				end[thr_i] = -1;
		}
		thread_number += thread_divid;*/		
        //printf("thread_number = %d\n", thread_number);	
		//printf("%d %d %d start[%d] = %d end[%d] = %d\n", thread_number, thread_number_prior, thread_number-thread_number_prior, thread_number, start[thread_number], thread_number-1, end[thread_number-1]);		
		thread_number_prior = thread_number;
		sum_number = 0; 
		thread_number_counter = 0;
		start[thread_number] = prior_column;
		if ( sum_5_number > 0 )
		{
		for ( i = prior_column; i < prior_column_n; i+=pack_i )
		{
			if ( flag[i] )
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column-1 && k > prior_column_prior_level-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else if ( k >= prior_column )
					{
						seperator_in_column_2[i] = j - offset_U[i];
						break;
					}
					else continue;
				}
				for ( j = offset_U[i+1]; j < offset_U[i+2]; j++ )
				{
					k = row_ptr_U[j];
					if (  k <= prior_column-1 && k > prior_column_prior_level-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else if ( k >= prior_column )
					{
						seperator_in_column_2[i+1] = j - offset_U[i+1];
						break;
					}
					else continue;
				}
				if ( sum_number >= sum_5_number/num_thread || i+1 == prior_column_n-1 )
				{
					end[thread_number] = i+1;
					sign[thread_number] = 2;
					if ( sum_4_number == 0 ) assign[thread_number] = 1;
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+2;
					sum_number = 0;
				}
				pack_i = 2;
			}
			else
			{
				for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
				{
					k = row_ptr_U[j];
					if ( k <= prior_column-1 && k > prior_column_prior_level-1 )
					{
						sum_number += offset_L[k+1] - offset_L[k] - 1;
					}
					else if ( k >= prior_column )
					{
						seperator_in_column_2[i] = j - offset_U[i];
						break;
					}
					else continue;
				}
				if ( sum_number >= sum_5_number/num_thread || i == prior_column_n-1 )
				{
					end[thread_number] = i;
					sign[thread_number] = 2;
					if ( sum_4_number == 0 ) assign[thread_number] = 1;
					thread_number++;
					thread_number_counter++;
					start[thread_number] = i+1;
					sum_number = 0;
				}
				pack_i = 1;
			}		 
		}
		}
	
	    else
		{
			/*int temp_val = (prior_column_n-prior_column)/num_thread;
			for ( i = prior_column+temp_val; i < prior_column_n; i+=temp_val )
			{
				end[thread_number] = i;
				thread_number++;
				thread_number_counter++;
				start[thread_number] = i+1;
			}
			end[thread_number-1] = prior_column_n-1;*/
			for ( i = prior_column; i < prior_column_n; i++ )
			{
				seperator_in_column_2[i] = seperator_in_column[i];
			}

			/*	start[thread_number] = prior_column;
				end[thread_number] = prior_column_n - 1;
				assign[thr_i] = 1;
				sign[thread_number] = 2;

			for ( thr_i = thread_number+1; thr_i < thread_number+num_thread; thr_i++ )*/
			for ( thr_i = thread_number; thr_i < thread_number+num_thread; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
				sign[thr_i] = 2;
			}
			thread_number += num_thread;
			thread_number_counter += num_thread;
		}
		
		if ( thread_number_counter != num_thread )
		{
			thread_number_cha = num_thread - thread_number_counter;
			for ( thr_i = thread_number; thr_i < thread_number+thread_number_cha; thr_i++ )
			{
				start[thr_i] = 0;
				end[thr_i] = -1;
				sign[thr_i] = 2;
			}
			thread_number += thread_number_cha;
		}
		//printf("2-thread_number_counter = %d thread_number = %d\n", thread_number_counter, thread_number);
		start[thread_number] = prior_column;
		end[thread_number] = prior_column_n-1;
		if ( sum_5_number == 0 && sum_4_number == 0 ) assign[thread_number] = 1;
		sign[thread_number] = 0;
		thread_number++;
	}
	
	for ( i = thread_number; i < thread_number+num_thread; i++ )
	{
		start[i] = 0;
		end[i] = -1;
		// if ( sign[i] == 1 )
		// {
			// printf("start[%d] = %d end[%d] = %d sign[%d] = %d\n", i, start[i], i, end[i], i, sign[i]);
		// }
	}
	thread_number += num_thread-1;
	/*for ( i = thread_number_prior_level; i < thread_number; i++ )
	{
		// if ( sign[i] == 0 )
		// {
			printf("start[%d] = %d end[%d] = %d sign[%d] = %d end-start+1 = %d\n", i, start[i], i, end[i], i, sign[i], end[i]-start[i]+1);
		// }
	}*/
	// end[thread_number_prior_level-1] = 3399999;
	
	/*for ( i = level_var; i < level_var_end; i++ )
	{
		// start[i] = end[i-1]+1;
		// end[i] = start[i] + 2; 
		// if ( sign[i] == 0 )
		{
			printf("start[%d] = %d end[%d] = %d sign[%d] = %d end-start+1 = %d\n", i, start[i], i, end[i], i, sign[i], end[i]-start[i]+1);
		}
	}*/
	// for ( i = prior_column_c; i < prior_column_n; i++ )
	// {
	// 	printf("sep_c[%d] = %d sep_c_2[%d] = %d\n", i, seperator_in_column[i], i, seperator_in_column_2[i]);
	// }
	printf("%d ~ %d has %lf computations in square ratio = %lf%%\n", prior_column, next_column, sum_4_number, sum_4_number / sum_5_number *100);
	printf("%d ~ %d has %lf computations in tri ratio = %lf%%\n", prior_column, next_column, sum_5_number - sum_4_number, (sum_5_number - sum_4_number) / sum_5_number*100);
	printf("%d ~ %d has %lf computations in all\n", prior_column, next_column, sum_5_number);
	/* printf("%d ~ %d has %d prior columns\n", prior_column, next_column, sum_flag); */
	printf("all the columns in square are divided into %d part\n", thread_number);

  double *L, *U, *xx;
  double U_diag;
  int current_column;
  
  L = ( double * )_mm_malloc(sizeof(double) * lnz, 64);
  U = ( double * )_mm_malloc(sizeof(double) * unz, 64);
  xx = ( double *)_mm_malloc(sizeof(double) * n, 64);

  for ( i = 0; i < n; i++ )
  {
	//xx[i] = 0;
    L[offset_L[i]] = 1.0;
  }

    double *x_result, *x_real_result;
    int error_result = 0;
    double t_s, t_e;
    
    NicsLU_Solve(solver, b, x);

	int loop = atoi(argv[6]);
    double sum_time = 0;	

    double** xx1 = malloc(sizeof(double*)* num_thread);
	double** xx2 = malloc(sizeof(double*)* num_thread);
	double** dv1 = malloc(sizeof(double*)* num_thread);
	double** dv2 = malloc(sizeof(double*)* num_thread);

 #pragma omp parallel for
 for(int i=0; i<num_thread; i++){
    xx1[i] = ( double *)_mm_malloc(sizeof(double) *2* n, 64); 
	memset(xx1[i], 0, sizeof(double) * 2*n);
	xx2[i] = xx1[i] + n;
	 // xx2[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	 // memset(xx2[i], 0, sizeof(double) * n);  
	// xx_next_column = xx;
	dv1[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv1[i], 0, sizeof(double) * 4096);
	dv2[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv2[i], 0, sizeof(double) * 4096);
	
}

	double t11, t12;
	double sum_tt = 0;
	
/* 	for ( i = 0; i < thread_number; i++ )
	{
		printf("start[%d] = %d end[%d] = %d\n", i, start[i], i, end[i]);
	} */
	
	//printf("Time of %d ~ %d columns = %lf\n", prior_column_c, n, t12);
	//int thread_number_2 = atoi(argv[12]);
	printf("thread_number = %d thread_number_prior_level = %d\n", thread_number, thread_number_prior_level);
	printf("loop = %d\n", loop);
	int thresold = atoi(argv[3]);
	bitInt *tag = (bitInt *)_mm_malloc(sizeof(bitInt) * (n/4 + 1),64);
		// bitInt *tag = (bitInt *)malloc(sizeof(bitInt) * (n/4 + 1));

// MyUnion *tag = (bitInt *)_mm_malloc(sizeof(bitInt) * (n/4 + 1),64);
//	memset(tag, 0, sizeof(int) * n);
 
    printf("print: %d %d sizeof(bitInt) = %d n/4+1 = %d\n", prior_column_c, next_column, sizeof(bitInt), n/4+1);
	int xa_trans_2[2];
	xa_trans_2[0] = 0;
	xa_trans_2[1] = next_column - prior_column_c;
	int *asub_U_level_2 = (int *)malloc(sizeof(int) * (next_column-prior_column_c));
	for ( i = 0; i < next_column-prior_column_c; i++ )
	{
		asub_U_level_2[i] = prior_column_c + i;
	}

	/*double *y, *x_me;
	y = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
	x_me = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
	double *x_real = (double *)malloc(sizeof(double) * n);*/


	for ( jj = 0; jj < loop; jj++ )
	{

		memset(L,0, sizeof(double) * lnz );
		memset(U,0, sizeof(double) * unz );

 #pragma omp parallel for
 for(int i=0; i<num_thread; i++){
	memset(xx1[i], 0, sizeof(double) * 2*n);
	xx2[i] = xx1[i] + n; 
	 // xx2[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	 // memset(xx2[i], 0, sizeof(double) * n);  
	// xx_next_column = xx;
    memset(dv1[i], 0, sizeof(double) * 4096);
    memset(dv2[i], 0, sizeof(double) * 4096);
	
}


		for ( i = 0; i < n; i++ )
  		{
			//xx[i] = 0;
  	  		L[offset_L[i]] = 1.0;
  		}
        /*for ( i = 0; i < unz; i++ )
		{
			if ( U[i] != 0 && lx[i] == 0 ) printf("U[%d] = %lf nicslu_U[%d] = %lf\n", i, U[i], i, lx[i]);
		}*/

		// for (i = 0; i < nnz; ++i) ax[i] *= (double)rand() / RAND_MAX * 2.;

		t_s = microtime();
		/*memset(U, 0, sizeof(double) * unz);
		memset(L, 0, sizeof(double) * lnz);
		  for ( i = 0; i < n; i++ )
		{
			//xx[i] = 0;
			L[offset_L[i]] = 1.0;
		}*/
		#pragma omp parallel for
		for(i=0; i<(n/4+1); i++)
		{
		    tag[i].bit32 = 0;
		}
		// memset(tag, 0, sizeof(int) * n);

        // if ( jj == 5 )
		// for ( i = 0; i < (n/4+1); i++ )
		// {
		// 	if ( tag[i].boolvec[0] != 0 || tag[i].boolvec[1] != 0 || tag[i].boolvec[2] != 0 || tag[i].boolvec[3] != 0 )
		// 		printf("main: jj = %d - %d: %d %d %d %d\n", jj, i, tag[i].boolvec[0], tag[i].boolvec[1], tag[i].boolvec[2], tag[i].boolvec[3]);
		// }

		for ( i = 0; i < thread_number_prior_level; i+=num_thread )
		// for ( i = 0; i < max_level+1; i++ )
		{
			x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_sum_final, flag, x, num_thread, i+num_thread, start, end, L, U, xx1, xx2, dv1, dv2, i, asub_U_level, ux, lx, tag, 0, 0, xa_trans);
		}
		x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, num_thread, i+num_thread, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, i, asub_U_level, seperator_in_column_2, ux, lx, tag, sum_level+1, max_level+1, xa_trans);

		//  x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, num_thread, i+num_thread, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, i, asub_U_level_2, seperator_in_column_2, ux, lx, tag, 0, 1, xa_trans_2);

		 x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, prior_column_c, next_column, num_thread, i+num_thread, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, (thread_number-thread_number_prior_level-1)/num_thread, asub_U_level, seperator_in_column_2, sign, ux, lx, thread_number_prior_level, thread_number, tag, assign);

		t_e = microtime() - t_s;
		sum_time += t_e;
		
		

		/*NicsLU_ReFactorize(solver, ax, 0);
		NicsLU_Solve(solver, b, x);

		for ( i = 0; i < n; i++ )
		{
			y[i] = 1.0;
		}

		for ( i = 0; i < n; i++ )
		{
			for ( j = offset_L[i]+1; j < offset_L[i+1]; j++ )
			{
				y[row_ptr_L[j]] -= y[i] * L[j];
			}
		}

		for ( i = 0; i < n; i++ )
		{
			x_me[i] = y[i];
		}

		x_me[n-1] = y[n-1]/U[unz-1];
		for ( i = n-1; i > 0; i-- )
		{
			for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
			{
				x_me[row_ptr_U[j]] -= x_me[i] *U[j];
			}
			x_me[i-1] = x_me[i-1]/U[offset_U[i]-1];
		}   

		memset(x_real, 0, sizeof(double) * n);
		int error_lu_gp = 0;
		int error_nic = 0;
		for ( i = 0; i < n; i++ ) x_real[row_perm_inv[i]] = x_me[i];
		for ( i = 0; i < n; i++ )
		{
			if ( fabs(x[i]-x_real[i]) > 10 )
			{
				error_nic++;
				printf("nicslu[%d] = %lf me[%d] = %lf\n", i, x[i], i, x_real[i]);
			}	
		}
    	printf("error of x results are: %d\n", error_nic);*/

	     printf("Time of All columns: %lf\n", t_e);
		
		/*for(int i=0; i<num_thread; i++){
       xx2[i] = xx1[i] + n;
       for ( k = 0; k < n; k++ )
       {
         if ( xx1[i][k] != 0 || xx2[i][k] != 0 ) printf("Error in xx! 000!\n");
       }

       for ( k = 0; k < 4096; k++ )
       {
         if ( dv1[i][k] != 0 || dv2[i][k] != 0 ) printf("Error in dv! 000!\n");
       }
	}*/

	}

	// for ( i = 0; i < unz; i++ )
	// {
	// 	printf("U[%d] = %lf nic_U[%d] = %lf\n", i, U[i], i, lx[i]);
	// }

	printf("Average Time of LU_GP_ssn_column_storage_multi_row_computing: %lf\n", sum_time/loop);
	// printf("thread_number_p_l = %d thread_number = %d\n", thread_number_prior_level, thread_number);
	// for ( j = 0; j < loop; j++ )
	// {
	// 	t_s = microtime();
	// 	for ( i = thread_number_prior_level; i < thread_number; i+=num_thread )
	// 	{
	// 		x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, prior_column_c, next_column, num_thread, i+num_thread, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, i, asub_U_level, seperator_in_column_2, sign, ux, lx, thread_number);
	// 	}
	// 	t_e = microtime() - t_s;
	// 	printf("NEXT-Time of %d~%d columns: %g\n", prior_column_c, next_column, t_e);
	// }

  printf("solve: 000000000000000000000000000000000000000\n");
  int error_lu_gp = 0;
  int error_nic = 0;

 /* double *y, *x_me;
//   y = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
//   x_me = ( double *)_mm_malloc( sizeof( double ) * n, 64 );

    y = ( double *)malloc( sizeof( double ) * n);
    x_me = ( double *)malloc( sizeof( double ) * n);

	 printf("solve: 11111111111111111111111111111111111111111111111\n");

  for ( i = 0; i < n; i++ )
  {
	  y[i] = 1.0;
  }

   

  for ( i = 0; i < n; i++ )
  {
	  for ( j = offset_L[i]+1; j < offset_L[i+1]; j++ )
	  {
		  y[row_ptr_L[j]] -= y[i] * L[j];
	  }
  }

  printf("solve: 2222222222222222222222222222222222222222222222\n");

  //x[n-1] = y[n-1];
  for ( i = 0; i < n; i++ )
  {
	  x_me[i] = y[i];
  }

  x_me[n-1] = y[n-1]/U[unz-1];
  for ( i = n-1; i > 0; i-- )
  {
	  for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
	  {
		  x_me[row_ptr_U[j]] -= x_me[i] *U[j];
	  }
	  x_me[i-1] = x_me[i-1]/U[offset_U[i]-1];
  }   
  
  double *x_real = (double *)malloc(sizeof(double) * n);
  memset(x_real, 0, sizeof(double) * n);
  for ( i = 0; i < n; i++ ) x_real[row_perm_inv[i]] = x_me[i];
  for ( i = 0; i < n; i++ )
  {
	//   printf("nicslu[%d] = %lf me_x[%d] = %lf\n", i, x[i], i, x_real[i]);
	  if ( fabs(x[i]-x_real[i]) > 0.1 )
	  {
		  error_nic++;
		//   printf("nicslu[%d] = %lf me[%d] = %lf\n", i, x[i], i, x_real[i]);
	  }	
	  // if ( fabs(lu_gp_x[i]-x_real[i]) > 0.1 )
	  // {
		  // error_lu_gp++;
		  //printf("lu_gp[%d] = %lf me[%d] = %lf\n", i, lu_gp_x[i], i, x_real[i]);
	  // }	
  }
    printf("error of x results are: %d\n", error_nic);*/

    error_nic = 0;
	for ( i = 0; i < unz; i++ )
	{
	//   printf("nicslu[%d] = %lf me[%d] = %lf\n", i, lx[i], i, U[i]);
	  if ( fabs(lx[i]-U[i]) > 0.1 )
	  {
		  error_nic++;
		//   printf("nicslu[%d] = %lf me_U[%d] = %lf\n", i, lx[i], i, U[i]);
	  }	
	}
	printf("error of U results are: %d\n", error_nic); 

	error_lu_gp = 0;
	for ( i = 0; i < lnz; i++ )
	{
	//   printf("nicslu[%d] = %lf me[%d] = %lf\n", i, lx[i], i, U[i]);
	  if ( fabs(ux[i]-L[i]) > 0.1 )
	  {
		  error_lu_gp++;
		   printf("nicslu[%d] = %lf me_L[%d] = %lf\n", i, ux[i], i, L[i]);
	  }
	}
	printf("error of L results are: %d\n", error_lu_gp); 

    // for ( i = 0; i < loop; i++ )
    // {
	//     t_s = microtime();
	//     x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, prior_column_c, next_column, num_thread, thread_number, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, thread_number_prior_level, asub_U_level, seperator_in_column_2, sign, ux, lx);
	// 	x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, prior_column_c, next_column, num_thread, thread_number, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, thread_number_prior_level, asub_U_level, seperator_in_column_2, sign, ux, lx);
    //    	t_e = microtime() - t_s;
    //     printf("Time of %d~%d columns: %g\n", prior_column_c, next_column, t_e);
	//     sum_time += t_e;
    // }
    // printf("Average Time of LU_GP_ssn_column_storage_multi_row_computing: %g\n", sum_time/loop); 
 
EXIT:
    free(ax);
    free(ai);
    free(ap);
    free(b);
    free(x);
    NicsLU_Free(solver);
    return 0;
}
