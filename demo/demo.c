/*this demo is the same as demo.cpp except for that this demo calls the C interface*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00

#ifdef _WIN32
#pragma comment(lib, "nicslu.lib")
#endif

double microtime()
{
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv,&tz);

	return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

const char *const ORDERING_METHODS[] = { "", "", "", "", "AMD", "AMM", "AMO1","AMO2","AMO3", };

int main(int argc[], char *argv[])
{
    int ret;
    _double_t *ax = NULL, *b = NULL, *x = NULL;
    _uint_t *ai = NULL, *ap = NULL;
    _uint_t n, row, col, nz, nnz, i, j;
    _handle_t solver = NULL;
    _double_t res[4], cond, det1, det2, fflop, sflop;
    size_t mem;
    _double_t *cfg;
    const _double_t *stat;

    /*print license*/
    //PrintNicsLULicense(NULL);

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
    ReadMatrixMarketFile(argv[1], &row, &col, &nz, ax, ai, ap, NULL, NULL, NULL);
    printf("******************************Matrix %s: row %d, col %d, nnz %d\n", argv[1], n, n, nnz);

    /*read RHS B*/
    b = (_double_t *)malloc(sizeof(_double_t)*n);
    ReadMatrixMarketFile(argv[1], &row, &col, &nz, b, NULL, NULL, NULL, NULL, NULL);
    for ( i = 0; i < n; i++ ) b[i] = 1.0;
    x = (_double_t *)malloc(sizeof(_double_t)*n);
    memset(x, 0, sizeof(_double_t) * n);

    /*initialize solver*/
    ret = NicsLU_Initialize(&solver, &cfg, &stat, NULL);
    if (__FAIL(ret))
    {
        printf("Failed to initialize, return = %d\n", ret);
        goto EXIT;
    }
    //printf("NICSLU version %.0lf\n", stat[31]);
    cfg[0] = 1.; /*enable timer*/
   // cfg[3] = 4;

    /*pre-ordering (do only once)*/
    NicsLU_Analyze(solver, n, ax, ai, ap, MATRIX_ROW_REAL, NULL, NULL, NULL, NULL);
    //printf("analysis time: %g\n", stat[0]);
    printf("best ordering method: %s\n", ORDERING_METHODS[(int)stat[16]]);

    /*create threads (do only once)*/
    NicsLU_CreateThreads(solver, 0); /*use all physical cores*/

    /*factor & solve (first-time)*/
    NicsLU_FactorizeMatrix(solver, ax, 0); /*use all created threads*/
    //printf("factor time: %g\n", stat[1]);
    NicsLU_Solve(solver, b, x);
    //printf("solve time: %g\n", stat[2]);

   // SparseResidual(n, ax, ai, ap, b, x, res, MATRIX_ROW_REAL);
    //printf("residual RMSE: %g\n", res[0]);

    /*Now we have finished a factorization and a solving.
    In many applications like circuit simulation, 
    we need to solve the linear system many times with different values (but the symbolic pattern keeps unchanged).
    The following code simulates such a case.*/
    
    printf("sn of NICSLU is: %lf\n", stat[12]);
    double time = 0;
    int th = atoi(argv[2]);
    int loop = atoi(argv[3]);
    double t1, t2;

    for (j = 0; j < loop; ++j) /*do 5 iterations*/
    {
        /*matrix and RHS values change*/
        //for (i = 0; i < nnz; ++i) ax[i] *= (_double_t)rand() / RAND_MAX * 2.;
        //for (i = 0; i < n; ++i) b[i] *= (_double_t)rand() / RAND_MAX * 2.;

        /*factor & solve again based on the changed matrix & RHS*/
        //NicsLU_FactorizeMatrix(solver, ax, 0); /*use all created threads*/
	t1 = microtime();
	NicsLU_ReFactorize(solver, ax, th);
	t2 = microtime() - t1;
//        printf("re-factor [%d] time: %g\n", j + 1, stat[1]);
//        printf("re-factor [%d] time by me: %g\n", j + 1, t2);
	time += t2;
        //NicsLU_Solve(solver, b, x);
        //printf("solve [%d] time: %g\n", j + 1, stat[2]);

        //SparseResidual(n, ax, ai, ap, b, x, res, MATRIX_ROW_REAL);
        //printf("residual [%d] RMSE: %g\n", j + 1, res[0]);
    }

   // NicsLU_Solve(solver, b, x);
   /* for ( i = 0; i < n; i++ )
    {
	    printf("x[%d] = %g\n", i, x[i]);
    }*/
    printf("Average time of refact is: %g\n", time/loop);

    /*finally, print some statistical information*/
    printf("nnz(factors): %.0lf\n", stat[8]); //# of factors
    printf("nnz of LU = %lf\n", stat[9]+stat[10]);

    //NicsLU_Flops(solver, 1, &fflop, &sflop);
    //printf("factor flops: %e, solving flops: %e\n", fflop, sflop); /*# of flops*/

    //NicsLU_ConditionNumber(solver, ax, &cond); /*condition number estimation*/
    //printf("condest: %g\n", cond);

    //NicsLU_Determinant(solver, &det1, &det2); /*determinant*/
    //printf("determinant: %.16lf x 10^(%.0lf)\n", det1, det2);

    //NicsLU_MemoryUsage(solver, &mem); /*get virtual memory usage*/
    //printf("memory usage: %.0lf KB\n", (double)(mem >> 10));

    //NicsLU_DrawFactors(solver, "add20.bmp", 512);

EXIT:
    free(ax);
    free(ai);
    free(ap);
    free(b);
    free(x);
    NicsLU_Free(solver);
#ifdef _WIN32
    getchar();
#endif
    return 0;
}
