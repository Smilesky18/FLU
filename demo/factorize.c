#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
# include <math.h>

int FLU_Symbolic(int n, int k, int *pinv, int *stack, int *flag, int *pend, int *appos, int *uindex, int *llen, void *lu, size_t *up, int *aidx, int arownnz)
{
	int top;
	int i, col, j, jnew;
	int head, pos;
	int *uidx;
	int ucol;
	int unnz;

	
}

int FLU_Factorize(int n, int *ap, int *ai, double *ax, int *perm_r, int *perm_c, int lu_est)
{
    int i, j, k, current_column, start, end;
    double *xx = (double *)malloc(sizeof(double) * n);
    memset(xx, 0, sizeof(double) * n);
    double *L = (double *)malloc(sizeof(double) * lu_est);
    double *U = (double *)malloc(sizeof(double) * lu_est);
    int *row_ptr_L = (int *)malloc(sizeof(int) * lu_est);
    int *row_ptr_U = (int *)malloc(sizeof(int) * lu_est);
    int *offset_L = (int *)malloc(sizeof(int) * n+1);
    int *offset_U = (int *)malloc(sizeof(int) * n+1);
    int top;
    int *pinv = (int *)malloc(sizeof(int) * n);
    memset(pinv, -1, sizeof(int) * n);
    int *stack = (int *)malloc(sizeof(int) * n);
    int *flag = (int *)malloc(sizeof(int) * n);
    int *pend = (int *)malloc(sizeof(int) * n);
    int *appos = (int *)malloc(sizeof(int) * n);
    int *llen = (int *)malloc(sizeof(int) * n);
    int *ulen = (int *)malloc(sizeof(int) * n);

    for ( i = 0; i < n; i++ )
    {
        current_column = perm_c[i];
        start = ap[current_column];
        end = ap[current_column+1];

        top = FLU_Symbolic(n, i, pinv, stack, flag, pend, appos, \
			(uint__t *)(((byte__t *)lu_array) + up[i]), ulen, lu_array, up, &ai[start], end-start);

		for ( j = ap[current_column]; j < ap[current_column+1]; j++ )
		{
			xx[ai[j]] = ax[j];
		}

		for ( k = top; k < n; k++ )
		{

		}

    }
}
