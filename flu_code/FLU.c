
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>
# include <math.h>
#include <sched.h>

#define MICRO_IN_SEC 1000000.00
/* Time Stamp */
double microtime()
{
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}


void FLU ( int *ai, int *ap, double *ax, int *row_ptr_L, int *offset_L, double *L, int *row_ptr_U, int *offset_U, double *U, int n, int nnz, int lnz, int unz, int *perm_c, int *perm_r, int thread, int loop)
{
    int i, j;
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
    if ( thread == 0 ) num_thread = 32;
    else num_thread = thread;
    int gp_level, pri_level, sn_sum_final;
    int sum_sn_col = 0;
	FLU_Detect_SuperNode(row_ptr_L, offset_L, sn_record, sn_number, sn_row_num, n, sn_column_start, sn_column_end, &sn_sum_final, &sum_sn_col);
	FLU_Dependency_Analysis(row_ptr_U, offset_U, asub_U_level, xa_trans, prior_column_c, num_thread, &gp_level, &pri_level);

    char *fl = (char *)malloc(sizeof(char) * n);
	memset(fl, 0, sizeof(char) * n);
	int sum_fl = 0;
    // int sum_non_fl = 0;
	int sum_sn_cols = 0;
	char *symbol = (char *)malloc(sizeof(char) * n);
	memset(symbol, 0, sizeof(char) * n);
    int pack_i;

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
				if ( sum_fl > 100 ) 
                {
                    symbol[j] = 1;
                }
				sum_fl = 0;
			}
			pack_i = 8;
		}
		else
		{
            /*for ( int k = offset_U[i]; k < offset_U[i+1]; k++ )
				{
					fl[row_ptr_U[k]] = 1;
				}
				for ( int k = offset_U[i+1]; k < offset_U[i+2]; k++ )
				{
					if ( fl[row_ptr_U[k]] ) 
					{
						sum_fl++;
						fl[row_ptr_U[k]] = 0;
					}
                    // else
                    // {
                    //     sum_non_fl++;
                    // }
				}
				if ( sum_fl > 100 ) 
                {
                    symbol[i] = 1;
                    pack_i = 2;
                }
                else
                {
                    pack_i = 1;
                }
				sum_fl = 0;*/

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

    // double *xx = (double *)_mm_malloc(sizeof(double) * n, 64);

    double** xx1 = malloc(sizeof(double*)* num_thread);
	double** xx2 = malloc(sizeof(double*)* num_thread);
	double** dv1 = malloc(sizeof(double*)* num_thread);
	double** dv2 = malloc(sizeof(double*)* num_thread);
    // double** xx = malloc(sizeof(double*)* num_thread);
//  #pragma omp parallel for
 for(int i=0; i<num_thread; i++){
    xx1[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx1[i], 0, sizeof(double) * n);
	xx2[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	memset(xx2[i], 0, sizeof(double) * n);  
    // xx[i] = ( double *)_mm_malloc(sizeof(double) * n, 64); 
	// memset(xx[i], 0, sizeof(double) * n);  
	dv1[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv1[i], 0, sizeof(double) * 4096);
	dv2[i] = ( double * )_mm_malloc(sizeof(double) * 4096, 64);
    memset(dv2[i], 0, sizeof(double) * 4096);
 }

    double t1, t2;
    double sum_t = 0;
    // int loop;

    char *tag = (char *)malloc(sizeof(char) * n);
    int thresold = 4;
    sum_t = 0;

    printf("lnz= %d unz = %d nnz = %d\n", lnz, unz, nnz);
    int lunnz = lnz + unz;
    if ( lunnz / nnz >= 10 )
    {   
        printf("Compact mode!\n");
        kmp_set_defaults("KMP_AFFINITY=compact");
        for ( i = 0; i < loop; i++ )
        {
            t1 = microtime();
            memset(tag, 0, sizeof(char) * n);

            flu_double_col_computing(ax, ai, ap, n, lnz, unz, perm_c, perm_r, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, tag, gp_level+1, 0, xa_trans, num_thread, symbol, sum_fl, n_new, sn_number,  sn_column_start, sn_column_end);

            t2 = microtime() - t1;
            sum_t += t2;
        //    printf("Iteration %d, Time of FLU is: %lf\n", i, t2);
        }
         printf("Average Time of FLU in compact is: %lf\n", sum_t/loop);
    }
    else
    {
        printf("Scatter mode!\n");
       kmp_set_defaults("KMP_AFFINITY=scatter");
        for ( i = 0; i < loop; i++ )
        {
            t1 = microtime();
            memset(tag, 0, sizeof(char) * n);

            flu_fact(ax, ai, ap, n, lnz, unz, perm_c, perm_r, row_ptr_L, offset_L, row_ptr_U, offset_U, sn_record, thresold, L, U, xx1, xx2, dv1, dv2, asub_U_level, tag, gp_level+1, 0, xa_trans, num_thread, sn_number,  sn_column_start, sn_column_end, sn_row_num);

            t2 = microtime() - t1;
            sum_t += t2;

//         printf("Iteration %d, Time of FLU is: %lf\n", i, t2);
        }
        printf("Average Time of FLU in scatter is: %lf\n", sum_t/loop);
}
}
