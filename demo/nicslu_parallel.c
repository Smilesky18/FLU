/*this demo is the same as demo.cpp except for that this demo calls the C interface*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
#define MICRO_IN_SEC 1000000.00

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
    _double_t *ax = NULL, *b = NULL, *x = NULL;
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

    /*initialize solver*/
    if (__FAIL(NicsLU_Initialize(&solver, &cfg, &stat, &last_err)))
    {
        printf("Failed to initialize\n");
        goto EXIT;
    }
    cfg[0] = 1.; /*enable timer*/
   // cfg[3] = 4;
    /*pre-ordering (do only once)*/
    NicsLU_Analyze(solver, n, ax, ai, ap, MATRIX_COLUMN_REAL, NULL, NULL, NULL, NULL);

    /*create threads (do only once)*/
    NicsLU_CreateThreads(solver, 0); /*use all physical cores*/
    /*factor & solve (first-time)*/
    NicsLU_FactorizeMatrix(solver, ax, 0); /*use all created threads*/
    printf("Time of fact: %g\n", stat[1]);


    printf("n = %d nnz = %d\n", n, nnz);
    printf("lnz = %d unz = %d\n", (int)stat[10], (int)stat[9]);
    printf("***********SuperNode************\n");
    printf("sn of NICSLU is: %lf\n", stat[12]);
	int num_th = atoi(argv[2]);
	printf("num_th = %d\n", num_th);
	int loop = atoi(argv[3]);
	 double time_fact_start, time_fact_end;
	   double sum_time_nic = 0;
         for ( i = 0; i < loop; i++ )
	 {
	// time_fact_start = microtime();
         NicsLU_ReFactorize(solver, ax, num_th); 
	// time_fact_end = microtime() - time_fact_start;
	 //printf("Time of re-fact by mine: %g\n", time_fact_end);
	 printf("Time of re-fact: %g\n", stat[1]);
	 sum_time_nic += stat[1];
	 }
	 printf("Average Time of nicslu: %g\n", sum_time_nic/loop);
	 /*printf("Number of factorizations executed = %lf\n", stat[14]);
	 printf("Number of re-factorizations executed = %lf\n", stat[15]);*/

	/*NicsLU_Solve(solver, b, x);
	for ( i = 0; i < n; i++ )
    	{
            printf("x[%d] = %g\n", i, x[i]);
    	}*/

    /*finally, print some statistical information*/

EXIT:
    free(ax);
    free(ai);
    free(ap);
    free(b);
    free(x);
    NicsLU_Free(solver);
    return 0;
}
