
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

if ( atoi(argv[5]) )
{
    // struct mc64_control control;
    // struct mc64_info info;
	// mc64_default_control(&control);
	// int job = 5;
	// int matrix_type = 2;
	// int *perm_mc64 = (int *) malloc((n+n)*sizeof(int));
    // double *scale = (double *) malloc((n+n)*sizeof(double));
	
	// mc64_matching(job, matrix_type, n, n, ap, ai, ax, &control, &info, perm_mc64, scale);

	// if(info.flag<0) {
    //      printf("Failure of mc64_matching with info.flag=%d", info.flag);
    //   }
	
	// printf("Row permutation\n");
    // for(i=0; i < n; i++) 
    // {
    //     if ( perm_mc64[i] != i )
    //         printf("row permuted! %8d", perm_mc64[i]);
    // }
    // printf("\nColumn permutation\n");

    // int *ap_mc64 = (int *)malloc(sizeof(int) * n);
	// int *ai_mc64 = (int *)malloc(sizeof(int) * nnz);
	// int *col_len = (int *)malloc(sizeof(int) * n);
	// double *ax_mc64 = (double *)malloc(sizeof(double) * nnz);
	// int p;

	// for ( i = 0; i < n; i++ )
	// {
	// 	col_len[perm_mc64[i+n]] = ap[i+1] - ap[i];
	// }
	// ap_mc64[0] = 0;
	// for ( i = 1; i < n+1; i++ )
	// {
	// 	ap_mc64[i] = ap_mc64[i-1] + col_len[i-1];
	// }
	// for ( i = 0; i < n; i++)
	// {
	// 	int col_in_mc64 = perm_mc64[i+n];

	// 	for ( j = ap_mc64[col_in_mc64], p = ap[i]; j < ap_mc64[col_in_mc64+1], p < ap[i+1]; j++, p++ )
	// 	{
	// 		ai_mc64[j] = perm_mc64[ai[p]];
	// 		ax_mc64[j] = ax[p];
	// 	}
	// }
    // /*for(i=n; i<n+n; i++) 
    // {
    //     if ( perm_mc64[i] != i )
    //         printf("column permuted! %8d", perm_mc64[i]);
    // }*/

	// // CSR READ & a+a' 
	// for ( int i = 0; i < nnz; i++ ) 
	// {
	// 	// ax_csr[i] = ax[i];
	// 	// ai_csr[i] = ai[i];

    //     ax_csr[i] = ax_mc64[i];
	// 	ai_csr[i] = ai_mc64[i];
	// }
	// for ( int i = 0; i < n+1; i++ )
	// {
	// 	// ap_csr[i] = ap[i];

    //     ap_csr[i] = ap_mc64[i];
	// }
	// SparseTranspose(n, ax_csr, ai_csr, ap_csr, 0);
	// printf("Tranpose matrix success!\n");
	// char *flag = (char *)malloc(sizeof(char) * n);
	// memset(flag, 0, sizeof(char) * n);
	// int *aat_ap = (int *)malloc(sizeof(int) * n+1);
	// int *aat_ai_temp = (int *)malloc(sizeof(int) * nnz*2);
	// aat_ap[0] = 0;
	// int len = 0;
	// int pack_j, pack_p;
	// /*for ( i = 0; i < n; i++ )
	// {
	// 	for ( j = ap[i], p = ap_csr[i]; j < ap[i+1] && p < ap_csr[i+1]; j+=pack_j, p+=pack_p )
	// 		{
	// 			if ( ai[j] < ai_csr[p] && ai[j] != i )
	// 			{
	// 				aat_ai_temp[len++] = ai[j];
	// 				pack_j = 1;
	// 				pack_p = 0;
	// 			}
	// 			else if ( ai[j] == ai_csr[p] && ai[j] != i )
	// 			{
	// 				aat_ai_temp[len++] = ai[j];
	// 				pack_j = 1;
	// 				pack_p = 1;
	// 			}
	// 			else if ( ai[j] > ai_csr[p] && ai_csr[p] != i )
	// 			{
	// 				aat_ai_temp[len++] = ai_csr[p];
	// 				pack_j = 0;
	// 				pack_p = 1;
	// 			}
	// 			else if ( ai[j] == i )
	// 			{
	// 				pack_j = 1;
	// 				pack_p = 0;
	// 			}
	// 			else
	// 			{
	// 				pack_j = 0;
	// 				pack_p = 1;
	// 			}	
	// 		}
	// 	for ( ; j < ap[i+1]; j++ ) 
	// 	{
	// 		if ( ai[j] != i )
	// 			aat_ai_temp[len++] = ai[j];
	// 	}
	// 	for ( ; p < ap_csr[i+1]; p++ )
	// 	{
	// 		if ( ai_csr[p] != i )
	// 			aat_ai_temp[len++] = ai_csr[p];
	// 	} 
	// 	aat_ap[i+1] = len;
	// }
	// */
    // for ( i = 0; i < n; i++ )
	// {
	// 	for ( j = ap_mc64[i], p = ap_csr[i]; j < ap_mc64[i+1] && p < ap_csr[i+1]; j+=pack_j, p+=pack_p )
	// 		{
	// 			if ( ai_mc64[j] < ai_csr[p] && ai_mc64[j] != i )
	// 			{
	// 				aat_ai_temp[len++] = ai_mc64[j];
	// 				pack_j = 1;
	// 				pack_p = 0;
	// 			}
	// 			else if ( ai_mc64[j] == ai_csr[p] && ai_mc64[j] != i )
	// 			{
	// 				aat_ai_temp[len++] = ai_mc64[j];
	// 				pack_j = 1;
	// 				pack_p = 1;
	// 			}
	// 			else if ( ai_mc64[j] > ai_csr[p] && ai_csr[p] != i )
	// 			{
	// 				aat_ai_temp[len++] = ai_csr[p];
	// 				pack_j = 0;
	// 				pack_p = 1;
	// 			}
	// 			else if ( ai_mc64[j] == i )
	// 			{
	// 				pack_j = 1;
	// 				pack_p = 0;
	// 			}
	// 			else
	// 			{
	// 				pack_j = 0;
	// 				pack_p = 1;
	// 			}	
	// 		}
	// 	for ( ; j < ap_mc64[i+1]; j++ ) 
	// 	{
	// 		if ( ai_mc64[j] != i )
	// 			aat_ai_temp[len++] = ai_mc64[j];
	// 	}
	// 	for ( ; p < ap_csr[i+1]; p++ )
	// 	{
	// 		if ( ai_csr[p] != i )
	// 			aat_ai_temp[len++] = ai_csr[p];
	// 	} 
	// 	aat_ap[i+1] = len;
	// }
	
    // int *aat_ai = (int *)malloc(sizeof(int) * len);
	// for ( i = 0; i < n; i++ )
	// {
	// 	for ( j = aat_ap[i]; j < aat_ap[i+1]; j++ )
	// 	{
	// 		aat_ai[j] = aat_ai_temp[j];
	// 	}
	// }
	// printf("a+a' permute success!\n");

	// // MEITS 
	// int n_csr =  n;
	// int *perm = (int *)malloc(sizeof(int) * n);
	// int *iperm = (int *)malloc(sizeof(int) * n);
    // int options[10];
    // options[0] = 0;
    // int numflag = 0;

	// printf("In METIS!\n");

	// METIS_NodeND(&n_csr, aat_ap, aat_ai, &numflag, options, perm, iperm); // metis 4.0
    // // METIS_NodeND(&n_csr, aat_ap, aat_ai, NULL, NULL, perm, iperm); // metis 5.0

	// int *perm_mc64_inv_row = (int *)malloc(sizeof(int) * n);
	// int *perm_mc64_inv_col = (int *)malloc(sizeof(int) * n);

	// for ( i = 0; i < n; i++ )
	// {
	// 	perm_mc64_inv_row[perm_mc64[i]] = i;
	// }
	// for ( i = 0; i < n; i++ )
	// {
	// 	perm_mc64_inv_col[perm_mc64[i+n]] = i;
	// }

	// for ( int i = 0; i < n; i++ )
	// {
	// 	row_perm[i] = perm_mc64_inv_row[perm[i]];
	// 	col_perm[i] = perm_mc64_inv_col[perm[i]];
	// 	// if ( perm[i] != iperm[i] ) printf("perm[%d] = %d iperm[%d] = %d\n", i, perm[i], i, iperm[i]);
	// }
    // cfg[3] = 1;
}

    cfg[0] = 1.; /*enable timer*/
	// cfg[3] = 0; // 0: no ordering; 1: user ordering; 2:selects the best one among all the built-in methods in parallel; 3: selects the best one among all the built-in methods in sequential; 4: AMD; 5: AMM; 6: AMO1; 7: AMO2; 8: AMO3;
    // cfg[8] = 4;
    /*pre-ordering (do only once)*/
	int ordering = 1;
	int *r_perm = (int *)malloc(sizeof(int) * n);
	int *c_perm = (int *)malloc(sizeof(int) * n);
	// FLU_Analyze(n, ap, ai, ordering);
    NicsLU_Analyze(solver, n, ax, ai, ap, MATRIX_COLUMN_REAL, NULL, NULL, NULL, NULL);
	// printf("Estimated: lnz + unz -n = %lf\n", stat[4]);

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
		// printf("stat[11] = %lf stat[13] = %lf stat[14] = %lf stat[15] = %lf\n", stat[11], stat[13], stat[14], stat[15]);
    }
    // printf("Actual: lnz = %lf unz = %lf  nnz(A) = %d lnz+unz-n = %lf\n", stat[10], stat[9], nnz, stat[10]+stat[9]-n);
	printf("Average time of FACT is: %lf\n", t_fact/loop_2);
    // printf("cfg[8] = %lf cfg[9] = %lf\n", cfg[8], cfg[9]);

	/*lx = (_double_t *)malloc(sizeof(_double_t) * stat[9]);
    ux = (_double_t *)malloc(sizeof(_double_t) * stat[10]);
	NicsLU_GetFactors(solver, lx, li, lp, ux, ui, up, sort, row_perm, col_perm, row_scale, col_scale);

	NicsLU_Solve(solver, b, x);
	for ( i = 0; i < n; i++ )
	{
			// printf("%d\n", row_perm[i]);
			// printf("%lf\n", x[i]);
	}


	for (i = 0; i < n; i++) col_perm_inv[col_perm[i]] = i;
    for (i = 0; i < n; i++) row_perm_inv[row_perm[i]] = i;*/

	/*int lnz = (int)stat[10]; // Number of non-zeros in L 
    int unz = (int)stat[9];
	printf("************value in L: ");
	for ( i = 0; i < lnz; i++ )
	{
			// printf("%d\n", row_perm[i]);
			printf("%lf ", ux[i]);
	}
	printf("\n************value in U: ");
	for ( i = 0; i < unz; i++ )
	{
			// printf("%d\n", row_perm[i]);
			printf("%lf ", lx[i]);
	}*/

    /*lx = (_double_t *)malloc(sizeof(_double_t) * stat[9]);
    ux = (_double_t *)malloc(sizeof(_double_t) * stat[10]);
    li = (_uint_t *)malloc(sizeof(_uint_t) * stat[9]);
    ui = (_uint_t *)malloc(sizeof(_uint_t) * stat[10]);
    NicsLU_GetFactors(solver, lx, li, lp, ux, ui, up, sort, row_perm_inv, col_perm_inv, row_scale, col_scale);
    for ( i = 0; i < n; i++ )
    {
        if ( row_perm[i] != row_perm_inv[i] || col_perm[i] != col_perm_inv[i] )
            printf("Pivot occurs in FACT!\n");
    }*/

// 	char fi2[20] = "lu.bmp";
// 	NicsLU_DrawFactors(solver, fi2, 2048);

    /* Get L/U structural information from NicSLU */
    lx = (_double_t *)malloc(sizeof(_double_t) * stat[9]);
    ux = (_double_t *)malloc(sizeof(_double_t) * stat[10]);
    li = (_uint_t *)malloc(sizeof(_uint_t) * stat[9]);
    ui = (_uint_t *)malloc(sizeof(_uint_t) * stat[10]);
    NicsLU_GetFactors(solver, lx, li, lp, ux, ui, up, sort, row_perm, col_perm, row_scale, col_scale);
    for (i = 0; i < n; i++) col_perm_inv[col_perm[i]] = i;
    for (i = 0; i < n; i++) row_perm_inv[row_perm[i]] = i;

	for ( i = 0; i < n; i++ )
	{
		// if ( row_perm[i] != col_perm[i] )
			// printf("r_perm[%d] = %d row_perm_inv[%d] = %d c_perm[%d] = %d col_perm_inv[%d] = %d\n", i, r_perm[i], i, row_perm_inv[i], i, c_perm[i], i, col_perm_inv[i]);
	}
    
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
		// printf("sn_offset[%d] = %d\n", i, sn_offset[i]);
	}
	double *sn_value = (double *)malloc(sizeof(double) * sn_offset[sn_sum_final]); 
	printf("sn_offset[sn_sum_final] = %d\n", sn_offset[sn_sum_final]);

	// int n_after_double_column = FLU_double_computing(flag, argv[7], row_ptr_U, offset_U, n, asub_U_level, row_ptr_U_after_double_column);
	// printf("gp_level = %d\n", gp_level);
	// printf("xa_trans[0] = %d\n", xa_trans[0]);
	// for ( j = xa_trans[pri_level]; j < xa_trans[gp_level+1]; j++ )
	// {
	// 	printf("col = %d\n", asub_U_level[j]);
	// }
	char *fl = (char *)malloc(sizeof(char) * n);
	memset(fl, 0, sizeof(char) * n);
	int sum_fl = 0;
	int sum_sn_cols = 0;
	for ( i = 0; i < n; i++ )
	{
		if ( sn_number[i] >= 0 )
		{
				sum_sn_cols += offset_U[i+1] - offset_U[i] - 1;
				// fl[i] = 1;
		}
		// printf("\n**************col %d************\n", i);
		for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
		{
			if ( sn_number[row_ptr_U[j]] >= 0 )//&& sn_number[i] == -1 )
			{
				fl[i] = 1;
				// sum_fl++;
				break;
			}
		}
	}
	/*int *sn_stat = (int *)malloc(sizeof(int) * sn_sum_final);
	memset(sn_stat, 0, sizeof(int) * sn_sum_final);
	for ( i = 0; i < n; i++ )
	{
		if ( fl[i] )
		{
			for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
			{
				if ( sn_number[row_ptr_U[j]] >= 0 )
				{
					sn_stat[sn_number[row_ptr_U[j]]]++;
				}
			}
			printf("\n*****************col %d***************\n", i);
			for ( j = 0; j < sn_sum_final; j++ )
			{
				if ( sn_stat[j] > 0 )
				{
					printf("%d ", sn_stat[j]);
					sn_stat[j] = 0;
				}
			}
		}
	}*/
	// int *sn_row = (int *)malloc(sizeof(int) * sn_sum_final+1);
	// memset(sn_row, 0, sizeof(int) * sn_sum_final+1);
	// int *xa = (int *)malloc(sizeof(int) * n);
	// memset(xa, 0, sizeof(int) * n);
	// int sum_sn_col_in_U = 0;
	// int *asub = (int *)malloc(sizeof(int) * n);
	// int *value = (int *)malloc(sizeof(int) * n);
	// memset(value, 0, sizeof(int) * n);
	// int *col_ptr = (int *)malloc(sizeof(int) * n);
	// memset(col_ptr, 0, sizeof(int) * n);
	// int *col_ptr_U = (int *)malloc(sizeof(int) * n);
	// memset(col_ptr_U, 0, sizeof(int) * n);
	// int *offset_U_row = (int *)malloc(sizeof(int) * n);
	// memset(offset_U_row, 0, sizeof(int) * n);
	// int *offset_U_row_2 = (int *)malloc(sizeof(int) * n);
	// memset(offset_U_row_2, 0, sizeof(int) * n);
	// int pack_k;
	// for ( int k = 0; k < n; k+=pack_k )
	// {
	// 	if ( sn_number[k] >= 0 )
	// 	{
	// 		int sum = xa[sn_number[k]];
	// 		int s = 0;

	// 		for ( i = k; i < k+8; i++ )
	// 		{
	// 			for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
	// 			{
	// 				value[row_ptr_U[j]]++;
	// 			}
	// 		}
	// 		/*for ( i = 0; i < n; i++ )
	// 		{
	// 			offset_U_row[i+1] = offset_U_row[i] + value[i];
	// 			offset_U_row_2[i+1] = offset_U_row_2[i] + value[i];
	// 		}
	// 		for ( i = 0; i < 8; i++ )
	// 		{
	// 			for ( j = offset_U[i+k]; j < offset_U[i+k+1]-1; j++ )
	// 			{
	// 				col_ptr_U[offset_U_row_2[row_ptr_U[j]]++] = i;
	// 			}
	// 		}*/
	// 		for ( i = 0; i < n; i++ )
	// 		{
	// 			if ( value[i] )
	// 			{
	// 				asub[sum] = i;
	// 				// temp[i] = sum;
	// 				sum++;
	// 				// col_ptr[sum] = col_ptr[sum-1] + value[i];
	// 				value[i] = 0;
	// 				// col_ptr_U[sum] += value[i];
	// 			}
	// 		}
	// 		// sn_row[sn_number[k]] = sum;
	// 		xa[sn_number[k]+1] = sum;
	// 		// for ( i = k; i < k+8; i++ )
	// 		// {
	// 		// 	for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
	// 		// 	{
	// 		// 		be[col_ptr_U[temp[row_ptr_U[j]]++] = i; 
	// 		// 	}
	// 		// }
			
	// 		// sum_sn_col_in_U += offset_U[i+1] - offset_U[i];
	// 		// for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
	// 		// {
	// 		// 	if ( sn_number[row_ptr_U[j]] < 0 )//&& sn_number[i] == -1 )
	// 		// 	{
	// 		// 		sum_fl++;
	// 		// 		// break;
	// 		// 	}
	// 		// }
	// 		// printf("\n%d %d %d %d", i, offset_U[i+1] - offset_U[i], row_ptr_U[offset_U[i+1]-1], row_ptr_L[offset_L[i+1]-1]);
	// 		// printf("\n***************col %d***************\n", i);
	// 		// for ( j = offset_U[i]; j < offset_U[i+1]; j++ )
	// 		// {
	// 		// 	// printf("%d ", sn_number[row_ptr_U[j]]);
	// 		// 	printf("%d ", row_ptr_U[j]);
	// 		// 	// for ( int k = offset_L[row_ptr_U[j]]; k < offset_L[row_ptr_U[j]+1]; k++ )
	// 		// 	// {
	// 		// 	// 	printf("%d ", row_ptr_L[k]);
	// 		// 	// }
	// 		// }
	// 		pack_k = 8;
	// 	}
	// 	else
	// 	{
	// 		pack_k = 1;
	// 	}
		
	// }
	// for ( int k = 0; k < n; k+=pack_k )
	// {
	// 	if ( sn_number[k] >= 0 )
	// 	{
	// 		for ( i = k; i < k+8; i++ )
	// 		{
	// 			for ( j = offset_U[i]; j < offset_U[i+1]-1; j++ )
	// 			{
	// 				value[row_ptr_U[j]]++;
	// 			}
	// 		}
	// 		pack_k = 8;
	// 	}
	// 	else
	// 	{
	// 		pack_k = 1;
	// 	}
		
	// }
	// for ( int k = 0; k < n; k+=pack_k )
	// {
	// 	if ( sn_number[k] >= 0 )
	// 	{
	// 		for ( i = 0; i < n; i++ )
	// 		{
	// 			offset_U_row[i+1] = offset_U_row[i] + value[i];
	// 			offset_U_row_2[i+1] = offset_U_row_2[i] + value[i];
	// 		}
	// 		for ( i = 0; i < 8; i++ )
	// 		{
	// 			for ( j = offset_U[i+k]; j < offset_U[i+k+1]-1; j++ )
	// 			{
	// 				col_ptr_U[offset_U_row_2[row_ptr_U[j]]++] = i;
	// 			}
	// 		}
	// 		pack_k = 8;
	// 	}
	// 	else
	// 	{
	// 		pack_k = 1;
	// 	}	
	// }
	// printf("unz = %d sum_sn_col_in_U = %d\n", unz, sum_sn_col_in_U);
	// printf("n = %d sum_fl = %d\n", n, sum_fl);
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

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_sn_row_computing(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end, sn_offset, sn_value, counter);

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_full_sn_computing_panel(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end, xa, asub, offset_U_row, col_ptr_U);

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_full_sn_computing(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end, fl);

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_dense_sn(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end);

		lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, 0, xa_trans, num_thread, sn_number, sn_column_start, sn_column_end);

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, pri_level+1, 0, xa_trans, num_thread, sn_number);

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, pri_level+1, xa_trans, num_thread, sn_number);

		// lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next_double_computing(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, xa_trans, num_thread, flag, n_after_double_column, row_ptr_U_after_double_column);

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
