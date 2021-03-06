
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
    _uint_t n, row, col, nz, nnz, i, j, k;
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
    NicsLU_Analyze(solver, n, ax, ai, ap, MATRIX_COLUMN_REAL, NULL, NULL, NULL, NULL);
    /*create threads (do only once)*/
    NicsLU_CreateThreads(solver, 0); /*use all physical cores*/
    /*factor(first-time)*/
	NicsLU_Factorize(solver, ax, 0);

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

    int *sn_col_offset = (int *)malloc(sizeof(int) * n+1);
    for ( i = 0; i < n+1; i++ )
    {
        sn_col_offset[i] = offset_L[i];
    }

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
    int prior_column_c = n;
	int *asub_U_level = (int *)malloc(sizeof(int) * prior_column_c);
	int *xa_trans = (int *)malloc(sizeof(int) * prior_column_c);
    int num_thread;
    if ( atoi(argv[2]) == 0 ) num_thread = 32;
	else num_thread = atoi(argv[2]);
    int gp_level, pri_level, sn_sum_final;
	FLU_Detect_SuperNode(row_ptr_L, offset_L, sn_record, sn_number, sn_row_num, n, sn_column_start, sn_column_end, &sn_sum_final);
	FLU_Dependency_Analysis(row_ptr_U, offset_U, asub_U_level, xa_trans, prior_column_c, num_thread, &gp_level, &pri_level);

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

    printf("sn_sum_final = %d\n", sn_sum_final);
    int *counter = (int *)malloc(sizeof(int) * 8);
	counter[0] = 0;
	counter[1] = 7;
	counter[2] = 13;
	counter[3] = 18;
	counter[4] = 22;
	counter[5] = 25;
	counter[6] = 27;
	counter[7] = 28;

    // int pack_i;
    int column_divid, sn_row;
   /* for ( i = 0; i < n; i+=pack_i )
    {
        if ( sn_number[i] >= 0 )
        {
            // printf("sn: %d\n", i);
            column_divid = sn_row_num[sn_number[i]] % 16;
            sn_row = sn_row_num[sn_number[i]];

            // j = offset_L[i];
            // row_ptr_L[j] = i+1;
            // row_ptr_L[j+1] = i+2;
            // row_ptr_L[j+2] = i+3;
            // row_ptr_L[j+3] = i+4;
            // row_ptr_L[j+4] = i+5;
            // row_ptr_L[j+5] = i+6;
            // row_ptr_L[j+6] = i+7;
            // for ( j = offset_L[i]+7, k = offset_L[i+7]+1; j < offset_L[i]+7+column_divid, k < offset_L[i+7]+1+column_divid; j++, k++ )
            // {
            //     row_ptr_L[j] = row_ptr_L[k];
            // }
            // for ( j = offset_L[i]+28+column_divid*8, k = offset_L[i+7]+1+column_divid; j < offset_L[i]+28+column_divid*8+sn_row-column_divid, k < offset_L[i+7]+1+column_divid+sn_row-column_divid; j++, k++ )
            // {
            //     row_ptr_L[j] = row_ptr_L[k];
            // }
            offset_L[i+1] = offset_L[i] + 7 + column_divid;
            offset_L[i+2] = offset_L[i+1] + 6 + column_divid;
            offset_L[i+3] = offset_L[i+2] + 5 + column_divid;
            offset_L[i+4] = offset_L[i+3] + 4 + column_divid;
            offset_L[i+5] = offset_L[i+4] + 3 + column_divid;
            offset_L[i+6] = offset_L[i+5] + 2 + column_divid;
            offset_L[i+7] = offset_L[i+6] + 1 + column_divid;

            pack_i = 8;
        }
        else
        {
            pack_i = 1;
        }
    }*/

    int *sn_offset = (int *)malloc(sizeof(int) * sn_sum_final+1);
    sn_offset[0] = 0;
    int sn_lp;
    for ( i = 0; i < sn_sum_final; i++ )
    {
        sn_lp = sn_row_num[i]*8 + 28;
        sn_offset[i+1] = sn_offset[i] + sn_lp;
    }
    double *sn_value = (double *)malloc(sizeof(double) * sn_offset[sn_sum_final]);

    // double *xx = (double *)_mm_malloc(sizeof(double) * n, 64);
	double *L = (double *)_mm_malloc(sizeof(double) * lnz, 64);
    double *LL = (double *)_mm_malloc(sizeof(double) * lnz, 64);
	double *U = (double *)_mm_malloc(sizeof(double) * unz, 64);
 
	for ( i = 0; i < n; i++ )
	{
		L[offset_L[i]] = 1.0;
        // xx[i] = 0;
        // LL[offset_L[i]] = 1.0;
	}
    double** xx1 = malloc(sizeof(double*)* num_thread);
	double** xx2 = malloc(sizeof(double*)* num_thread);
	double** dv1 = malloc(sizeof(double*)* num_thread);
	double** dv2 = malloc(sizeof(double*)* num_thread);
    double** xx = malloc(sizeof(double*)* num_thread);
 #pragma omp parallel for
 for(int i=0; i<num_thread; i++){
    xx1[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx1[i], 0, sizeof(double) * n);
	xx2[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx2[i], 0, sizeof(double) * n);  
    xx[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx[i], 0, sizeof(double) * n);  
	dv1[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv1[i], 0, sizeof(double) * 4096);
	dv2[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv2[i], 0, sizeof(double) * 4096);
 }

    double t1, t2;
    double sum_t = 0;
    int loop = atoi(argv[3]);

    for (j = 0; j < 0; ++j) /*do 5 iterations*/
    {
	    t1 = microtime();
	    NicsLU_ReFactorize(solver, ax, atoi(argv[2]));
	    t2 = microtime() - t1;
	    sum_t += t2;
        printf("Time of NicSLU is: %lf\n", t2);
    }
    printf("Average Time of NicSLU is: %lf\n", sum_t/loop);
	
	// double *xx = ( double *)_mm_malloc(sizeof(double) * n, 64);
	// memset(xx, 0, sizeof(double) * n);
	// t1 = microtime();
	// // gp(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, L, U, xx);
	// t2 = microtime() - t1;
	// printf("GP time is: %lf\n", t2);

    char *tag = (char *)malloc(sizeof(char) * n);
    int thresold = 4;
    sum_t = 0;

    for ( i = 0; i < loop; i++ )
    {
        t1 = microtime();
        memset(tag, 0, sizeof(char) * n);
        lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end, counter, sn_row_num, xx, LL, sn_value, sn_offset, sn_col_offset);

        // lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_double_computing(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, xa_trans, num_thread, symbol, sum_fl, n_new);
        t2 = microtime() - t1;
        sum_t += t2;
        printf("Time of FLU is: %lf\n", t2);
    }
    printf("Average Time of FLU is: %lf\n", sum_t/loop);

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
		if ( fabs(x[i]-x_real[i]) > 0.1 )
		{
			error_nic++;
		}	
	}
    // printf("error of x results are: %d\n", error_nic);

 	/* compariosn of U */
    error_nic = 0;
	for ( i = 0; i < unz; i++ )
	{
        // printf("nicslu_U[%d] = %lf me[%d] = %lf\n", i, lx[i], i, U[i]);
	  if ( fabs(lx[i]-U[i]) > 0.1 )
	  {
		  error_nic++;
        //   printf("nicslu_U[%d] = %lf me[%d] = %lf\n", i, lx[i], i, U[i]);
	  }	
	}
	printf("error of U results are: %d\n", error_nic); 

    /* compariosn of L */
	error_lu_gp = 0;
	for ( i = 0; i < lnz; i++ )
	{
	  if ( fabs(ux[i]-L[i]) > 0.1 )
	  {
		  error_lu_gp++;
	  }
	}
	// printf("error of L results are: %d\n", error_lu_gp); 

EXIT:
    free(ax);
    free(ai);
    free(ap);
    free(b);
    free(x);
    NicsLU_Free(solver);
    return 0;
}
