
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
#include "metis.h"
#include "hsl_mc64d.h"

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

void bubble_sort(int *a, double *value, int n)
{
    int i, j;
	int temp;
	double val;

	if ( value == NULL )
	{
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
    else
	{
		for( i = 0; i < n-1; i++)
		{
			for( j = 0; j < n-1-i; j++)
			{
				if( a[j] > a[j+1] )
				{
					temp = a[j];
					a[j] = a[j+1];
					a[j+1] = temp;

					val = value[j];
					value[j] = value[j+1];
					value[j+1] = val;
				}
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

	struct mc64_control control;
    struct mc64_info info;
	mc64_default_control(&control);
	int job = 5;
	int matrix_type = 2;
	int *perm_mc64 = (int *) malloc((n+n)*sizeof(int));
    double *scale = (double *) malloc((n+n)*sizeof(double));
	
	mc64_matching(job, matrix_type, n, n, ap, ai, ax, &control, &info, perm_mc64, scale);

	if(info.flag<0) {
         printf("Failure of mc64_matching with info.flag=%d", info.flag);
      }
	
	printf("Row permutation\n");
    // for(i=0; i < n; i++) printf("%8d", perm_mc64[i]);
    printf("\nColumn permutation\n");
    // for(i=n; i<n+n; i++) printf("%8d", perm_mc64[i]);

    /* Transpose A by mc64 */
	/*int *ap_mc64 = (int *)malloc(sizeof(int) * n);
	int *ai_mc64 = (int *)malloc(sizeof(int) * nnz);
	int *col_len = (int *)malloc(sizeof(int) * n);
	double *ax_mc64 = (double *)malloc(sizeof(double) * nnz);
	int p;

	for ( i = 0; i < n; i++ )
	{
		col_len[perm_mc64[i+n]] = ap[i+1] - ap[i];
	}
	ap_mc64[0] = 0;
	for ( i = 1; i < n+1; i++ )
	{
		ap_mc64[i] = ap_mc64[i-1] + col_len[i-1];
	}
	for ( i = 0; i < n; i++)
	{
		int col_in_mc64 = perm_mc64[i+n];

		for ( j = ap_mc64[col_in_mc64], p = ap[i]; j < ap_mc64[col_in_mc64+1], p < ap[i+1]; j++, p++ )
		{
			ai_mc64[j] = perm_mc64[ai[p]];
			ax_mc64[j] = ax[p];
		}
		bubble_sort(&ai_mc64[ap_mc64[col_in_mc64]], &ax_mc64[ap_mc64[col_in_mc64]], ap[i+1]-ap[i]);
	}
	printf("\nmc64 permute success!\n");*/
	// for ( i = 0; i < n; i++ )
	// {
	// 	printf("\n***************MC64: col %d row index**************\n", i);
	// 	for ( j = ap_mc64[i]; j < ap_mc64[i+1]; j++ )
	// 	{
	// 		printf("%d ", ai_mc64[j]);
	// 	}
	// }
	/* CSR READ & a+a' */
	for ( int i = 0; i < nnz; i++ ) 
	{
		// ax_csr[i] = ax_mc64[i];
		// ai_csr[i] = ai_mc64[i];

		ax_csr[i] = ax[i];
		ai_csr[i] = ai[i];
	}
	for ( int i = 0; i < n+1; i++ )
	{
		// ap_csr[i] = ap_mc64[i];

		ap_csr[i] = ap[i];
	}
	SparseTranspose(n, ax_csr, ai_csr, ap_csr, 0);
	printf("Tranpose matrix success!\n");
	char *flag = (char *)malloc(sizeof(char) * n);
	memset(flag, 0, sizeof(char) * n);
	int *aat_ap = (int *)malloc(sizeof(int) * n+1);
	int *aat_ai_temp = (int *)malloc(sizeof(int) * nnz*2);
	aat_ap[0] = 0;
	int len = 0;
	int pack_j, pack_p, p;
	for ( i = 0; i < n; i++ )
	{
		for ( j = ap[i], p = ap_csr[i]; j < ap[i+1] && p < ap_csr[i+1]; j+=pack_j, p+=pack_p )
			{
				if ( ai[j] < ai_csr[p] && ai[j] != i )
				{
					aat_ai_temp[len++] = ai[j];
					pack_j = 1;
					pack_p = 0;
				}
				else if ( ai[j] == ai_csr[p] && ai[j] != i )
				{
					aat_ai_temp[len++] = ai[j];
					pack_j = 1;
					pack_p = 1;
				}
				else if ( ai[j] > ai_csr[p] && ai_csr[p] != i )
				{
					aat_ai_temp[len++] = ai_csr[p];
					pack_j = 0;
					pack_p = 1;
				}
				else if ( ai[j] == i )
				{
					pack_j = 1;
					pack_p = 0;
				}
				else
				{
					pack_j = 0;
					pack_p = 1;
				}	
			}
		for ( ; j < ap[i+1]; j++ ) 
		{
			if ( ai[j] != i )
				aat_ai_temp[len++] = ai[j];
		}
		for ( ; p < ap_csr[i+1]; p++ )
		{
			if ( ai_csr[p] != i )
				aat_ai_temp[len++] = ai_csr[p];
		} 
		aat_ap[i+1] = len;
	}
	int *aat_ai = (int *)malloc(sizeof(int) * len);
	for ( i = 0; i < n; i++ )
	{
		// bubble_sort(&aat_ai_temp[aat_ap[i]], NULL, aat_ap[i+1]-aat_ap[i]);
		for ( j = aat_ap[i]; j < aat_ap[i+1]; j++ )
		{
			aat_ai[j] = aat_ai_temp[j];
		}
	}
	/*for ( i = 0; i < n; i++ )
	{
		// len += ap[i+1] - ap[i];
		for ( j = ap[i]; j < ap[i+1]; j++ )
		{
			if ( ai[j] != i )
			{
				flag[ai[j]] = 1;
				aat_ai_temp[len++] = ai[j];
			}
		}
		for ( j = ap_csr[i]; j < ap_csr[i+1]; j++ )
		{
			if ( flag[ai_csr[j]] || ai_csr[j] == i ) continue;
			else
			{
				flag[ai_csr[j]] = 1;
				aat_ai_temp[len++] = ai_csr[j];
			}
		}
		aat_ap[i+1] = len;
		memset(flag, 0, sizeof(char) * n);
	}
	printf("a+a' phase success!\n");
	int *aat_ai = (int *)malloc(sizeof(int) * len);
	for ( i = 0; i < n; i++ )
	{
		bubble_sort(&aat_ai_temp[aat_ap[i]], NULL, aat_ap[i+1]-aat_ap[i]);
		for ( j = aat_ap[i]; j < aat_ap[i+1]; j++ )
		{
			aat_ai[j] = aat_ai_temp[j];
		}
	}*/
	printf("a+a' permute success!\n");
	// for ( i = 0; i < n; i++ )
	// {
	// 	printf("\n***************col %d row index**************\n", i);
	// 	for ( j = aat_ap[i]; j < aat_ap[i+1]; j++ )
	// 	{
	// 		printf("%d ", aat_ai[j]);
	// 	}
	// }
	/* MEITS */
	idx_t n_csr = (idx_t) n;
	idx_t *perm = (idx_t *)malloc(sizeof(idx_t) * n);
	idx_t *iperm = (idx_t *)malloc(sizeof(idx_t) * n);
	idx_t *my_xadj = (idx_t *)malloc(sizeof(idx_t) * n+1);
	printf("len = %d\n", len);
	idx_t *my_adjncy = (idx_t *)malloc(sizeof(idx_t) * len);
	idx_t Opt [20];
	Opt[0] = 0;
	idx_t *Opt2 = NULL;
	idx_t* zero = NULL;

	// idx_t options[METIS_NOPTIONS];
	// // options[0] = 0;
    // // METIS_SetDefaultOptions(options);
	// options[METIS_OPTION_DBGLVL] = 1;
	// options[METIS_OPTION_NUMBERING] = 0;
	// options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;


	// for ( i = 0; i < METIS_NOPTIONS; i++ )
	// {
	// 	printf("options[%d] = %d\n", i, options[i]);
	// }

	printf("In METIS!\n");

	/*for ( int i = 0; i < n+1; i++ )
	{
		my_xadj[i] = (idx_t)aat_ap[i];
		// printf("my_xadj[%d] = %d\n", i, my_xadj[i]);
	}
	for ( int i = 0; i < len; i++ )
	{
		my_adjncy[i] = (idx_t)aat_ai[i];
		// printf("my_adjncy[%d] = %d\n", i, my_adjncy[i]);
	}*/

	printf("In METIS-2! n_csr = %d\n", n_csr);

	// int *perm = (int *)malloc(sizeof(int) * n);
	// int *iperm = (int *)malloc(sizeof(int) * n);

	int ret = METIS_NodeND(&n_csr, aat_ap, aat_ai, NULL, NULL, perm, iperm);

	printf("ret = %d\n", ret);

	// for ( int i = 0; i < n; i++ )
	// {
	// 	printf("perm_METIS[%d] = %d\n", i, perm[i]);
	// }

	int *perm_mc64_inv_row = (int *)malloc(sizeof(int) * n);
	int *perm_mc64_inv_col = (int *)malloc(sizeof(int) * n);

	for ( i = 0; i < n; i++ )
	{
		perm_mc64_inv_row[perm_mc64[i]] = i;
	}
	for ( i = 0; i < n; i++ )
	{
		perm_mc64_inv_col[perm_mc64[i+n]] = i;
	}

	for ( int i = 0; i < n; i++ )
	{
		row_perm[i] = perm_mc64_inv_row[perm[i]];
		col_perm[i] = perm_mc64_inv_col[perm[i]];
		// printf("perm[%d] = %d iperm[%d] = %d\n", i, perm[i], i, iperm[i]);
	}

    cfg[0] = 1.; /*enable timer*/
	cfg[3] = 1;
    /*pre-ordering (do only once)*/
    NicsLU_Analyze(solver, n, ax, ai, ap, MATRIX_COLUMN_REAL, row_perm, col_perm, NULL, NULL);
	printf("\nEstimated: lnz + unz -n = %lf\n", stat[4]);

    /*create threads (do only once)*/
    NicsLU_CreateThreads(solver, 0); /*use all physical cores*/
    /*factor & solve (first-time)*/
    int thres = atoi(argv[4]);
	double t_fact = 0;
    for ( i = 0; i < 10; i++ )
    {
        //for (j = 0; j < nnz; ++j) ax[j] *= (_double_t)rand() / RAND_MAX * 2.;
    	NicsLU_Factorize(solver, ax, thres); /*use all created threads*/
		printf("FACT time: %lf\n", stat[1]);
		t_fact += stat[1];
		// printf("stat[11] = %lf stat[13] = %lf stat[14] = %lf stat[15] = %lf\n", stat[11], stat[13], stat[14], stat[15]);
    }
    printf("Actual: lnz = %lf unz = %lf\n", stat[10], stat[9]);
	printf("Average time of FACT is: %lf\n", t_fact/10);

	char fi2[20] = "lu.bmp";
	NicsLU_DrawFactors(solver, fi2, 2048);

    /* Get L/U structural information from NicSLU */
    lx = (_double_t *)malloc(sizeof(_double_t) * stat[9]);
    ux = (_double_t *)malloc(sizeof(_double_t) * stat[10]);
    li = (_uint_t *)malloc(sizeof(_uint_t) * stat[9]);
    ui = (_uint_t *)malloc(sizeof(_uint_t) * stat[10]);
    NicsLU_GetFactors(solver, lx, li, lp, ux, ui, up, sort, row_perm, col_perm, row_scale, col_scale);
    for (i = 0; i < n; i++) col_perm_inv[col_perm[i]] = i;
    for (i = 0; i < n; i++) row_perm_inv[row_perm[i]] = i;

	// for ( i = 0; i < n; i++ )
	// {
	// 	if ( row_perm_inv[i] != col_perm_inv[i] )
	// 		printf("row_perm_inv[%d] = %d col_perm_inv[%d] = %d\n", i, row_perm_inv[i], i, col_perm_inv[i]);
	// }
    
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
	// int prior_column_c = atoi(argv[7]);
	int prior_column_c = n;
	int *asub_U_level = (int *)malloc(sizeof(int) * prior_column_c);
	int *xa_trans = (int *)malloc(sizeof(int) * prior_column_c);
	int num_thread;
	if ( atoi(argv[2]) == 0 ) num_thread = 32;
	else num_thread = atoi(argv[2]);
	int gp_level;
	FLU_Detect_SuperNode(row_ptr_L, offset_L, sn_record, n);
	FLU_Dependency_Analysis(row_ptr_U, offset_U, asub_U_level, xa_trans, prior_column_c, num_thread, &gp_level);
	// printf("gp_level = %d\n", gp_level);
	// printf("xa_trans[0] = %d\n", xa_trans[0]);
	// for ( j = 0; j < gp_level+1; j++ )
	// {
	// 	printf("xa_trans[%d] = %d\n", j, xa_trans[j]);
	// }

    /* Run NicSLU code */
	/*double time = 0;
    int th2 = atoi(argv[2]);
    int loop2 = atoi(argv[3]);
    double t1, t2;
    for (j = 0; j < loop2; ++j) 
    {
        //for (i = 0; i < nnz; ++i) ax[i] *= (_double_t)rand() / RAND_MAX * 2.;
        //for (i = 0; i < n; ++i) b[i] *= (_double_t)rand() / RAND_MAX * 2.;
		t1 = microtime();
        NicsLU_ReFactorize(solver, ax, th2);
		t2 = microtime() - t1;
        // printf("Time of NicSLU is: %g\n", t2);
        time += t2;
    }
	printf("Average time of NicSLU is: %lf\n", time/loop2);*/

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

	// for ( i = 0; i < n; i++ )
	// {
	// 	if ( col_perm[i] == i )
	// 		printf("col_perm[%d] = %d\n", i, col_perm[i]);
	// } 

	for ( int jj = 0; jj < loop; jj++ )
	{
		t_s = microtime();

		memset(tag, 0, sizeof(char) * n);
		lu_gp_sparse_supernode_dense_column_computing_v5_multi_row_computing_prior_next(ax, ai, ap, n, lnz, unz, row_perm_inv, col_perm, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, ux, lx, tag, gp_level+1, xa_trans, num_thread);

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
