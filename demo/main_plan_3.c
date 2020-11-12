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
    // int col_thresold = atoi(argv[2]);
	int col_thresold = 4;
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
	int *sn_number_cou = (int *)malloc(sizeof(int) * n);
	memset(sn_number_cou, 0, sizeof(int) * n);
	double sum = 0; 

    /*for ( i = 0; i <= sn_sum; i++ )
    {
		if ( sn_end[i] - sn_start[i] >= col_thresold ) 
		{
			for ( j = sn_start[i]; j <= sn_end[i]; j++ )
			{
				sn_record[j] = sn_end[i] - j;
				sn_num_record[j] = sn_sum_final;
				sn_number_cou[j] = sn_end[i] - sn_start[i] + 1;
			}
			sn_column_start[sn_sum_final] = sn_start[i];
			sn_column_end[sn_sum_final] = sn_end[i];
			sn_sum_final++;
		}
    }*/

	int sn_div;
	// int thre_1 = atoi(argv[12]);
	// int thre_2 = atoi(argv[13]);
	int thre_1 = 8;
	int thre_2 = 4;
	int sn_j;

    for ( i = 0; i <= sn_sum; i++ )
    {
		if ( sn_end[i] - sn_start[i] >= col_thresold ) 
		{
			if ( sn_end[i] - sn_start[i] + 1 > thre_1 )
			{
				for ( j = sn_start[i]; j <= sn_end[i]; j+=thre_2 )
				{
					if ( j+thre_2 <= sn_end[i] )
					{
						for ( sn_j = 0; sn_j < thre_2; sn_j++ )
						{
							sn_record[j+sn_j] = thre_2 - 1 - sn_j;
						}
					}
					sn_column_start[sn_sum_final] = j;
					sn_column_end[sn_sum_final] = j+3;
					sn_sum_final++;
					sum += 4;
				}
			}
			else
			{
				for ( j = sn_start[i]; j <= sn_end[i]; j++ )
				{
					sn_record[j] = sn_end[i] - j;
					sn_num_record[j] = sn_sum_final;
					sn_number_cou[j] = sn_end[i] - sn_start[i] + 1;
				}
				sn_column_start[sn_sum_final] = sn_start[i];
				sn_column_end[sn_sum_final] = sn_end[i];
				sn_sum_final++;
			}
		}
    }

	printf("sn_sum_final = %d\n", sn_sum_final);

	int l, m;
	// int prior_column_c = atoi(argv[7]);
	int prior_column_c = n;
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
		// asub_U_level[ xa[length[i]] ] = i;
		// xa[length[i]]++;
	}

	int num_thread = atoi(argv[2]);
	int sum_level = max_level;
	int sum_more_than_16_columns = 0;
	for ( i = 0; i < max_level+1; i++ )
	{
		if ( level[i] < num_thread )
		{
			sum_level = i;
			break; 
		}
		else
		{
			sum_more_than_16_columns += level[i];
		}
	}
	printf("max_level = %d sum_level = %d\n", max_level, sum_level);

	char *no_wait = (char *)malloc(sizeof(char) * n);
	memset(no_wait, 0, sizeof(char) * n);
	int sum_wait = 0;
	int sum_whe_wait = 0;
	int j1, j2, j3, j4;
	int *wait_col_index = (int *)malloc(sizeof(int) * n);
	memset(wait_col_index, -1, sizeof(int) * n);
	char *wait_index = (char *)malloc(sizeof(char) * unz);
	memset(wait_index, 0, sizeof(char) * unz);

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
				// printf("main-0: i = %d j1 = %d\n", i, j1);
				j2 = asub_U_level[j1];
				wait_col_index[j2] = offset_U[j2+1] - offset_U[j2] - 1;
				for ( l = offset_U[j2+1]-2; l >= offset_U[j2]; l-- )
				{
					if ( no_wait[row_ptr_U[l]] == 0 )
					{
						sum_wait++;
						wait_col_index[j2] = l - offset_U[j2];
						wait_index[l] = 1;
						// if ( row_ptr_U[l] == 95273 )
							// printf("main-1: %d kk = %d k = %d l = %d\n", row_ptr_U[l], j1, j2, l); 
						// if ( sn_num_record[row_ptr_U[l]] != -1 )
							// printf("warning!: this is a sn!\n");
						// printf("depend col in U = %d\n", l-offset_U[j2]);
						// printf("depend col in U = %d\n", offset_U[j2+1] -1 - l);
					}
					// if ( row_ptr_U[l] == 95273 )
							// printf("main-2: %d kk = %d k = %d\n", row_ptr_U[l], j1, j2); 
					sum_whe_wait++;
				}
				// if (wait_col_index[j2] != offset_U[j2+1] - offset_U[j2] - 2 )
					//  printf("wait_col_index[%d] = %d\n", j2, wait_col_index[j2]);
				// printf("*************************: %d u_col = %d\n", offset_U[j2+1] - offset_U[j2] - 1, j2);
			}
			for ( j1 = i; j1 < i+num_thread; j1++ )
			{
				if ( asub_U_level[j1] == 95273 )
					printf("main-3: %d kk = %d k = %d i = %d\n", row_ptr_U[l], j1, asub_U_level[j1], i); 
				no_wait[asub_U_level[j1]] = 1;
			}
			// printf("i+num_th = %d\n", i+num_thread);
		}
		else
		{
			break;
		}
		// printf("asub_U_level[%d] = %d\n", i, asub_U_level[i]);
	}
	// i = i+1;
	for ( ; i < xa_trans[max_level+1]; i++ )
	{
		j2 = asub_U_level[i];
			for ( l = offset_U[j2+1]-2; l >= offset_U[j2]; l-- )
			{
				if ( no_wait[row_ptr_U[l]] == 0 )
				{
					sum_wait++;
					wait_col_index[j2] = l - offset_U[j2];
					wait_index[l] = 1;
				}
				sum_whe_wait++;
			}
	}


	// double thresold_2 = atof(argv[10]);
	int sum_flag = 0;
	int i_counter = 0;
	
	int prior_column = prior_column_c;

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

	int loop = atoi(argv[3]);
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

	printf("loop = %d\n", loop);
	// int thresold = atoi(argv[3]);
	int thresold = 4;
	// bitInt *tag = (bitInt *)_mm_malloc(sizeof(bitInt) * (n/4 + 1),64);
		char *tag = (char *)malloc(sizeof(char) * n);
		int *flag = (int *)malloc(sizeof(int) * 5);
		int *start = (int *)malloc(sizeof(int) * 5);
		int *end = (int *)malloc(sizeof(int) * 5);
		int *seperator_in_column = (int *)malloc(sizeof(int) * 5);
		int *seperator_in_column_2 = (int *)malloc(sizeof(int) * 5);
		int *sign = (int *)malloc(sizeof(int) * 5);
		int *assign = (int *)malloc(sizeof(int) * 5);
		int thread_number, thread_number_prior_level;
		int next_column;

	for ( int jj = 0; jj < loop; jj++ )
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


		t_s = microtime();

		// #pragma omp parallel for
		// for(i=0; i<(n/4+1); i++)
		// {
		//     tag[i].bit32 = 0;
		// }
		memset(tag, 0, sizeof(char) * n);

		x_result = lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, sn_num_record, sn_column_start, sn_column_end, sn_sum_final, flag, x, num_thread, i+num_thread, start, end, seperator_in_column, L, U, xx1, xx2, dv1, dv2, i, asub_U_level, seperator_in_column_2, ux, lx, tag, sum_level+1, max_level+1, xa_trans, wait_col_index, wait_index, no_wait, sn_number_cou);

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

	}

	printf("Average Time of LU_GP_ssn_column_storage_multi_row_computing: %lf\n", sum_time/loop);

  int error_lu_gp = 0;
  int error_nic = 0;

  double *y, *x_me;
//   y = ( double *)_mm_malloc( sizeof( double ) * n, 64 );
//   x_me = ( double *)_mm_malloc( sizeof( double ) * n, 64 );

    y = ( double *)malloc( sizeof( double ) * n);
    x_me = ( double *)malloc( sizeof( double ) * n);

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
    printf("error of x results are: %d\n", error_nic);

    error_nic = 0;
	for ( i = 0; i < unz; i++ )
	{
	//   printf("nicslu_U[%d] = %lf me_U[%d] = %lf\n", i, lx[i], i, U[i]);
	  if ( fabs(lx[i]-U[i]) > 0.1 )
	  {
		  error_nic++;
		//   printf("nicslu_U[%d] = %lf me_U[%d] = %lf\n", i, lx[i], i, U[i]);
	  }	
	}
	printf("error of U results are: %d\n", error_nic); 

	error_lu_gp = 0;
	for ( i = 0; i < lnz; i++ )
	{
	//   printf("nicslu_L[%d] = %lf me_L[%d] = %lf\n", i, ux[i], i, L[i]);
	  if ( fabs(ux[i]-L[i]) > 0.1 )
	  {
		  error_lu_gp++;
//		   printf("nicslu[%d] = %lf me_L[%d] = %lf\n", i, ux[i], i, L[i]);
	  }
	}
	printf("error of L results are: %d\n", error_lu_gp); 
	for ( i = 0; i < lnz; i++ )
	{
	//   printf("nicslu_L[%d] = %lf me_L[%d] = %lf\n", i, ux[i], i, L[i]);
	}

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
