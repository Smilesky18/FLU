
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
// #include "metis.h"
// #include "hsl_mc64d.h"

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

int max ( int a, int b )
{
    if ( a > b ) return a;
    return b;
}
 
int dump( int a, int *arr, int n )
{
	for (int i = 0; i < n; i++ )
	{
		if ( a != arr[i] )
		{
			continue;
		}
		else
		{
			return 1;
		}
	}
	return 0;
}

void bubble_sort(int *a, int n)
{
    int i, j;
	int temp;
    
    for( i = 0; i < n-1; i++)
    {
        for( j = 0; j < n-1-i; j++)
        {
            if( a[j] > a[j+1] )
            {
                temp = a[j];
                a[j] = a[j+1];
                a[j+1] = temp;
            }
        }
    }
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
	_double_t *ax_csr = NULL;
    _uint_t *ai = NULL, *ap = NULL;
	_uint_t *ai_csr = NULL, *ap_csr = NULL;
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
	ax_csr = (_double_t *)malloc(sizeof(_double_t)*nnz);
	ai_csr = (_uint_t *)malloc(sizeof(_uint_t)*nnz);
    ap_csr = (_uint_t *)malloc(sizeof(_uint_t)*(1 + n));
    
    row_perm = (_uint_t *)malloc(sizeof(_uint_t)*n);
    col_perm = (_uint_t *)malloc(sizeof(_uint_t)*n);
    col_perm_inv = (_uint_t *)malloc(sizeof(_uint_t)*n);
    row_perm_inv = (_uint_t *)malloc(sizeof(_uint_t)*n);
    // row_perm_inv = NULL;
    // col_perm_inv = NULL;
    row_scale = (_double_t *)malloc(sizeof(_double_t)*n);
    col_scale = (_double_t *)malloc(sizeof(_double_t)*n); 
    lp = (_size_t *)malloc(sizeof(_size_t)*(1+n));
    up = (_size_t *)malloc(sizeof(_size_t)*(1+n));
	
    ReadMatrixMarketFile(argv[1], &row, &col, &nz, ax, ai, ap, NULL, NULL, NULL); // CSC Read
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
    int thres = atoi(argv[4]);
	int loop_2 = atoi(argv[3]);
	double t1, t2;
	double t_fact = 0;
    for ( i = 0; i < 1; i++ )
    {
        //for (j = 0; j < nnz; ++j) ax[j] *= (_double_t)rand() / RAND_MAX * 2.;
		t1 = microtime();
    	NicsLU_Factorize(solver, ax, thres); /*use all created threads*/
		t2 = microtime() - t1;
		t_fact += t2;
		printf("FACT time: %lf\n", t2);
    }
	printf("Average time of FACT is: %lf\n", t_fact/loop_2);

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
	int *sn_number = (int *)malloc(sizeof(int) * n);	
    memset(sn_number, -1, sizeof(int) * n);
	int *sn_row_num = (int *)malloc(sizeof(int) * n);	
    memset(sn_row_num, -1, sizeof(int) * n);
	int *sn_column_start = (int *)malloc(sizeof(int) * n);
    memset(sn_column_start, 0, sizeof(int) * n);
    int *sn_column_end = (int *)malloc(sizeof(int) * n);
    memset(sn_column_end, 0, sizeof(int) * n);
	int *counter = (int *)malloc(sizeof(int) * 8);
	memset(counter, 0, sizeof(int) * 8);
	counter[7] = 0;
	counter[6] = 7;
	counter[5] = 13;
	counter[4] = 18;
	counter[3] = 22;
	counter[2] = 25;
	counter[1] = 27;
	counter[0] = 0;
	// int prior_column_c = atoi(argv[7]);
	int prior_column_c = n;
	int *asub_U_level = (int *)malloc(sizeof(int) * prior_column_c);
	int *xa_trans = (int *)malloc(sizeof(int) * prior_column_c);
	int *flag = (int *)malloc(sizeof(int) * n);
	int *row_ptr_U_after_double_column = (int *)malloc(sizeof(int) * n);
	memset(flag, 0, sizeof(int) * n);
	int num_thread;
	if ( atoi(argv[2]) == 0 ) num_thread = 32;
	else num_thread = atoi(argv[2]);
	int gp_level, pri_level, sn_sum_final;
	FLU_Detect_SuperNode(row_ptr_L, offset_L, sn_record, sn_number, sn_row_num, n, sn_column_start, sn_column_end, &sn_sum_final);
	FLU_Dependency_Analysis(row_ptr_U, offset_U, asub_U_level, xa_trans, prior_column_c, num_thread, &gp_level, &pri_level);

	int *sn_offset = (int *)malloc(sizeof(int) * sn_sum_final+1);
	memset(sn_offset, 0, sizeof(int) * sn_sum_final+1);
	int *sn_len = (int *)malloc(sizeof(int) * sn_sum_final);
	
	for ( i = 0; i < sn_sum_final; i++ )
	{
		sn_len[i] = 28 + sn_row_num[i] * 8;
	}
	for ( i = 0; i < sn_sum_final; i++ )
	{
		sn_offset[i+1] = sn_offset[i] + sn_len[i];
	}
	double *sn_value = (double *)malloc(sizeof(double) * sn_offset[sn_sum_final]); 
	memset(sn_value, 0, sizeof(double) * sn_offset[sn_sum_final]);
	printf("sn_offset[sn_sum_final] = %d\n", sn_offset[sn_sum_final]);

	char *fl = (char *)malloc(sizeof(char) * n);
	memset(fl, 0, sizeof(char) * n);
	int sum_fl = 0;
	int sum_sn_cols = 0;
	int pack_i;
	char *symbol = (char *)malloc(sizeof(char) * n);
	memset(symbol, 0, sizeof(char) * n);
	for ( i = 0; i < n; i+=pack_i )
	{
		if ( sn_number[i] >= 0 )
		{
			for ( j = i; j < i+8; j+=2 )
			{
				for ( int k = offset_U[j]; k < offset_U[j+1]; k++ )
				{
					fl[row_ptr_U[k]] = 1;
				}
				for ( int k = offset_U[j+1]; k < offset_U[j+2]; k++ )
				{
					if ( fl[row_ptr_U[k]] ) 
					{
						sum_fl++;
						fl[row_ptr_U[k]] = 0;
					}
				}
				if ( sum_fl > 100 ) symbol[j] = 1;
				sum_fl = 0;
			}
			pack_i = 8;
		}
		else
		{
			pack_i = 1;
		}	
	}
	sum_fl = 0;
	for ( i = 0; i < n; i+=pack_i )
	{
		if ( symbol[i] )
		{
			sum_fl++;
			pack_i = 2;
		}
		else
		{
			sum_fl++;
			pack_i = 1;
		}
	}
	printf("n = %d sum_fl = %d\n", n, sum_fl);
	int *n_new = (int *)malloc(sizeof(int) * sum_fl);
	int cou = 0;
	for ( i = 0; i < n; i+=pack_i )
	{
		if ( symbol[i] )
		{
			n_new[cou++] = i;
			pack_i = 2;
		}
		else
		{
			n_new[cou++] = i;
			pack_i = 1;
		}
		
	}
    // for ( )

    /* Run NicSLU code */
	double time = 0;
    int th2 = atoi(argv[2]);
    // int loop2 = atoi(argv[3]);
	int loop2 = atoi(argv[6]);
    double ts_nic, te_nic;
    for (j = 0; j < loop2; ++j) 
    {
        //for (i = 0; i < nnz; ++i) ax[i] *= (_double_t)rand() / RAND_MAX * 2.;
        //for (i = 0; i < n; ++i) b[i] *= (_double_t)rand() / RAND_MAX * 2.;
		ts_nic = microtime();
        NicsLU_ReFactorize(solver, ax, th2);
		te_nic = microtime() - ts_nic;
        printf("Time of NicSLU is: %g\n", te_nic);
        time += te_nic;
    }
	printf("Average time of NicSLU is: %lf\n", time/loop2);

    /* Run FLU code */
	double *L, *U;
	L = ( double * )_mm_malloc(sizeof(double) * lnz, 64);
	U = ( double * )_mm_malloc(sizeof(double) * unz, 64);
	for ( i = 0; i < n; i++ )
	{
		L[offset_L[i]] = 1.0;
	}
    double** xx1 = malloc(sizeof(double*)* num_thread);
	double** xx2 = malloc(sizeof(double*)* num_thread);
	double** dv1 = malloc(sizeof(double*)* num_thread);
	double** dv2 = malloc(sizeof(double*)* num_thread);
 #pragma omp parallel for
 for(int i=0; i<num_thread; i++){
    xx1[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx1[i], 0, sizeof(double) * n);
	xx2[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx2[i], 0, sizeof(double) * n);  
	dv1[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv1[i], 0, sizeof(double) * 4096);
	dv2[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv2[i], 0, sizeof(double) * 4096);
	
}
	double t_s, t_e;    
	int loop = atoi(argv[3]);
    double sum_time = 0;	

	printf("loop = %d\n", loop);
	printf("Number of nonzeros in factors = %d\n", lnz+unz);
	// int thresold = atoi(argv[3]);
	int thresold = 4;
	char *tag = (char *)malloc(sizeof(char) * n);

	for ( i = 0; i < 1; i++ )
	{
		printf("max_level = %d xa_trans[max_level/2] = %d\n", gp_level, xa_trans[gp_level/2]);
	} 
// for ( i = 0; i < lnz; i++ ) printf("L[%d] = %lf\n", i, L[i]);
    t_s = microtime();
	memset(tag, 0, sizeof(char) * n);
	// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end);
	t_e = microtime() - t_s;
	printf("Time of FLU is: %lf\n", t_e);

	// for ( i = 0; i < lnz; i++ ) printf("L[%d] = %lf\n", i, L[i]);

	for ( int jj = 0; jj < loop; jj++ )
	{
		t_s = microtime();

		memset(tag, 0, sizeof(char) * n);

		lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_double_computing(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, xa_trans, num_thread, symbol, sum_fl, n_new);

		t_e = microtime() - t_s;
		sum_time += t_e;
		
	    printf("Time of FLU is: %lf\n", t_e);
	}

	printf("Average Time of FLU is: %lf\n", sum_time/loop);
	printf("nnz(factors): %.0lf\n", stat[8]);

	NicsLU_Solve(solver, b, x);

	int error_lu_gp = 0;
	int error_nic = 0;

	double *y, *x_me;
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
  
    /* compariosn of X */
	double *x_real = (double *)malloc(sizeof(double) * n);
	memset(x_real, 0, sizeof(double) * n);
	for ( i = 0; i < n; i++ ) x_real[row_perm_inv[i]] = x_me[i];
	for ( i = 0; i < n; i++ )
	{
		//   printf("nicslu[%d] = %lf me_x[%d] = %lf\n", i, x[i], i, x_real[i]);
		if ( fabs(x[i]-x_real[i]) > 0.1 )
		{
			error_nic++;
			// printf("nicslu[%d] = %lf me[%d] = %lf\n", i, x[i], i, x_real[i]);
		}	
	}
    printf("error of x results are: %d\n", error_nic);

 	/* compariosn of U */
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

    /* compariosn of L */
	error_lu_gp = 0;
	for ( i = 0; i < lnz; i++ )
	{
	//   printf("nicslu_L[%d] = %lf me_L[%d] = %lf\n", i, ux[i], i, L[i]);
	  if ( fabs(ux[i]-L[i]) > 0.1 )
	  {
		  error_lu_gp++;
		//   printf("nicslu[%d] = %lf me_L[%d] = %lf\n", i, ux[i], i, L[i]);
	  }
	}
	printf("error of L results are: %d\n", error_lu_gp); 

EXIT:
    free(ax);
    free(ai);
    free(ap);
    free(b);
    free(x);
    NicsLU_Free(solver);
    return 0;
}
