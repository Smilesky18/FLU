/* -------------------------------------------------------------------- */
/*      Example program to show the use of the "PARDISO" routine        */
/*      on for unsymmetric linear systems                               */
/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      http://www.pardiso-project.org                                  */
/*                                                                      */
/*  (C) Olaf Schenk, Institute of Computational Science                 */
/*      Universita della Svizzera italiana, Lugano, Switzerland.        */
/*      Email: olaf.schenk@usi.ch                                       */
/* -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nicslu.h"
#include <sys/time.h>
#include "metis.h"
#include "hsl_mc64d.h"
#define MICRO_IN_SEC 1000000.00

/* Time Stamp */
double microtime()
{
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

/* PARDISO prototype. */
void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
void pardiso_chkvec     (int *, int *, double *, int *);
void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);


int main( int argc[], char *argv[] ) 
{
   
    /*int    n = 8;
    int    ia[ 9] = { 0, 4, 7, 9, 11, 12, 15, 17, 20 };
    int    ja[20] = { 0,    2,       5, 6, 
                         1, 2,    4,
                            2,             7,
                               3,       6,
                         1,
                            2,       5,    7,
                         1,             6,
                            2,          6, 7 };
    double  a[20] = { 7.0,      1.0,           2.0, 7.0, 
                          -4.0, 8.0,      2.0,
                                1.0,                     5.0,
                                     7.0,           9.0,
                          -4.0,
                                7.0,           3.0,      8.0,
                           1.0,                    11.0,
                               -3.0,                2.0, 5.0 };

    int      nnz = ia[n];*/

    int m1, n1, nnz1, ii;
    ReadMatrixMarketFile(argv[1], &m1, &n1, &nnz1, NULL, NULL, NULL, NULL, NULL, NULL); 

    double *a = (double *)malloc(sizeof(double)*nnz1);
    int *row_ptr = (int *)malloc(sizeof(int)*nnz1);
    int *offset = (int *)malloc(sizeof(int)*(1 + n1));

    ReadMatrixMarketFile(argv[1], &m1, &n1, &nnz1, a, row_ptr, offset, NULL, NULL, NULL); // CSC Read
    printf("***********%s: row %d, col %d, nnz %d\n", argv[1], n1, n1, nnz1);
    printf("here here here here here here!!!!!!!!!!!!!!!!!!\n");

    // for ( ii = 0; ii < nnz1; ii++ ) row_ptr[ii]+=1;
    // for ( ii = 0; ii < n1+1; ii++ ) offset[ii]+=1;

    int n = n1;
    int nnz = nnz1;
    int *offset_A = (int *)malloc(sizeof(int)*(1 + n));
    int *row_ptr_A = (int *)malloc(sizeof(int)*(nnz));

    for ( int ii = 0; ii < n+1; ii++ ) 
    {
        offset_A[ii] = offset[ii];
        // printf("ia[%d] = %d\n", ii, ia[ii]);
    }
    for ( int ii = 0; ii < nnz; ii++ ) 
    {
        row_ptr_A[ii] = row_ptr[ii];
        // printf("a[%d] = %lf\n", ii, a[ii]);
    }

    int      mtype = 11;       

    /* RHS and solution vectors. */
    // double   b[n], x[n], diag[n];
    double *b = (double *)malloc(sizeof(double) * n);
    double *x = (double *)malloc(sizeof(double) * n);
    double *diag = (double *)malloc(sizeof(double) * n);
    int      nrhs = 1;          /* Number of right hand sides. */

    /* Internal solver memory pointer pt,                  */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    /* or void *pt[64] should be OK on both architectures  */ 
    void    *pt[64];

    /* Pardiso control parameters. */
    int      iparm[64];
    double   dparm[64];
    int      solver;
    int      maxfct, mnum, phase, error, msglvl;

    /* Number of processors. */
    int      num_procs;

    /* Auxiliary variables. */
    char    *var;
    int       k;

    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */
    int *pardiso_perm;

   if ( atoi(argv[3]) )
   {
    int i, j;
    struct mc64_control control;
    struct mc64_info info;
	mc64_default_control(&control);
	int job = 5;
	int matrix_type = 2;
	int *perm_mc64 = (int *) malloc((n+n)*sizeof(int));
    double *scale = (double *) malloc((n+n)*sizeof(double));
	
	mc64_matching(job, matrix_type, n, n, offset, row_ptr, a, &control, &info, perm_mc64, scale);

	if(info.flag<0) {
         printf("Failure of mc64_matching with info.flag=%d", info.flag);
      }
	
	printf("Row permutation\n");
    for(i=0; i < n; i++) 
    {
        if ( perm_mc64[i] != i )
            printf("row permuted! %8d", perm_mc64[i]);
    }
    printf("\nColumn permutation\n");

    int *ap_mc64 = (int *)malloc(sizeof(int) * n);
	int *ai_mc64 = (int *)malloc(sizeof(int) * nnz);
	int *col_len = (int *)malloc(sizeof(int) * n);
	double *ax_mc64 = (double *)malloc(sizeof(double) * nnz);
	int p;

	for ( i = 0; i < n; i++ )
	{
		col_len[perm_mc64[i+n]] = offset[i+1] - offset[i];
	}
	ap_mc64[0] = 0;
	for ( i = 1; i < n+1; i++ )
	{
		ap_mc64[i] = ap_mc64[i-1] + col_len[i-1];
	}
	for ( i = 0; i < n; i++)
	{
		int col_in_mc64 = perm_mc64[i+n];

		for ( j = ap_mc64[col_in_mc64], p = offset[i]; j < ap_mc64[col_in_mc64+1], p < offset[i+1]; j++, p++ )
		{
			ai_mc64[j] = perm_mc64[row_ptr[p]];
			ax_mc64[j] = a[p];
		}
	}
    /*for(i=n; i<n+n; i++) 
    {
        if ( perm_mc64[i] != i )
            printf("column permuted! %8d", perm_mc64[i]);
    }*/

	// CSR READ & a+a' 
    double *ax_csr = (double *)malloc(sizeof(double)*nnz);
	int *ai_csr = (int *)malloc(sizeof(int)*nnz);
    int *ap_csr = (int *)malloc(sizeof(int)*(1 + n));
	for ( i = 0; i < nnz; i++ ) 
	{
		// ax_csr[i] = a[i];
		// ai_csr[i] = row_ptr[i];

        ax_csr[i] = ax_mc64[i];
		ai_csr[i] = ai_mc64[i];
	}
	for ( i = 0; i < n+1; i++ )
	{
		// ap_csr[i] = offset[i];

        ap_csr[i] = ap_mc64[i];
	}
	SparseTranspose(n, ax_csr, ai_csr, ap_csr, 0);
	printf("Tranpose matrix success!\n");
	char *flag = (char *)malloc(sizeof(char) * n);
	memset(flag, 0, sizeof(char) * n);
	int *aat_ap = (int *)malloc(sizeof(int) * n+1);
	int *aat_ai_temp = (int *)malloc(sizeof(int) * nnz*2);
	aat_ap[0] = 0;
	int len = 0;
	int pack_j, pack_p;
	/*for ( i = 0; i < n; i++ )
	{
		for ( j = offset[i], p = ap_csr[i]; j < offset[i+1] && p < ap_csr[i+1]; j+=pack_j, p+=pack_p )
			{
				if ( row_ptr[j] < ai_csr[p] && row_ptr[j] != i )
				{
					aat_ai_temp[len++] = row_ptr[j];
					pack_j = 1;
					pack_p = 0;
				}
				else if ( row_ptr[j] == ai_csr[p] && row_ptr[j] != i )
				{
					aat_ai_temp[len++] = row_ptr[j];
					pack_j = 1;
					pack_p = 1;
				}
				else if ( row_ptr[j] > ai_csr[p] && ai_csr[p] != i )
				{
					aat_ai_temp[len++] = ai_csr[p];
					pack_j = 0;
					pack_p = 1;
				}
				else if ( row_ptr[j] == i )
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
		for ( ; j < offset[i+1]; j++ ) 
		{
			if ( row_ptr[j] != i )
				aat_ai_temp[len++] = row_ptr[j];
		}
		for ( ; p < ap_csr[i+1]; p++ )
		{
			if ( ai_csr[p] != i )
				aat_ai_temp[len++] = ai_csr[p];
		} 
		aat_ap[i+1] = len;
	}*/
	
    
    for ( i = 0; i < n; i++ )
	{
		for ( j = ap_mc64[i], p = ap_csr[i]; j < ap_mc64[i+1] && p < ap_csr[i+1]; j+=pack_j, p+=pack_p )
			{
				if ( ai_mc64[j] < ai_csr[p] && ai_mc64[j] != i )
				{
					aat_ai_temp[len++] = ai_mc64[j];
					pack_j = 1;
					pack_p = 0;
				}
				else if ( ai_mc64[j] == ai_csr[p] && ai_mc64[j] != i )
				{
					aat_ai_temp[len++] = ai_mc64[j];
					pack_j = 1;
					pack_p = 1;
				}
				else if ( ai_mc64[j] > ai_csr[p] && ai_csr[p] != i )
				{
					aat_ai_temp[len++] = ai_csr[p];
					pack_j = 0;
					pack_p = 1;
				}
				else if ( ai_mc64[j] == i )
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
		for ( ; j < ap_mc64[i+1]; j++ ) 
		{
			if ( ai_mc64[j] != i )
				aat_ai_temp[len++] = ai_mc64[j];
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
		for ( j = aat_ap[i]; j < aat_ap[i+1]; j++ )
		{
			aat_ai[j] = aat_ai_temp[j];
		}
	}
	printf("a+a' permute success!\n");

	// MEITS 4.0
	int n_csr = n;
	int *perm = (int *)malloc(sizeof(int) * n);
	int *iperm = (int *)malloc(sizeof(int) * n);
    int options[10];
    options[0] = 0;
    int numflag = 0;

	printf("In METIS!\n");

	METIS_NodeND(&n_csr, aat_ap, aat_ai, &numflag, options, perm, iperm);

    int *row_perm = (int *)malloc(sizeof(int)*n);
    int *col_perm = (int *)malloc(sizeof(int)*n);
    pardiso_perm = (int *)malloc(sizeof(int) * n);
	for ( int i = 0; i < n; i++ )
	{
        pardiso_perm[i] = perm[i] + 1;
		// row_perm[i] = perm_mc64_inv_row[perm[i]];
		// col_perm[i] = perm_mc64_inv_col[perm[i]];
		// printf("perm[%d] = %d iperm[%d] = %d\n", i, perm[i], i, iperm[i]);
	}
   }

    else
    {
        pardiso_perm = &idum;
    }



/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters and initialize the solvers      */
/*     internal adress pointers. This is only necessary for the FIRST   */
/*     call of the PARDISO solver.                                      */
/* ---------------------------------------------------------------------*/
      
    error = 0;
    solver = 0; /* use sparse direct solver */
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

    if (error != 0)
    {
        if (error == -10 )
           printf("No license file found \n");
        if (error == -11 )
           printf("License is expired \n");
        if (error == -12 )
           printf("Wrong username or hostname \n");
         return 1;
    }
    else
        printf("[PARDISO]: License check was successful ... \n");
 

    /* Numbers of processors, value of OMP_NUM_THREADS */
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }
    iparm[2]  = num_procs;

    iparm[10] = 0; /* no scaling  */
    iparm[12] = 0; /* no matching */
    iparm[1] = 2; // 0-AMD; 2-METIS 4.1; 3-METIS-5.1 default = 2
    if ( atoi(argv[3] ))
        iparm[4] = 1; // 0 - pardiso's reordering; 1 - user define ordering
    
    maxfct = 1;         /* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 0;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */
    int i;


/* -------------------------------------------------------------------- */    
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */ 
    for (i = 0; i < n+1; i++) {
        offset_A[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        row_ptr_A[i] += 1;
    }

    /* Set right hand side to i. */
    for (i = 0; i < n; i++) {
        b[i] = 1;
    }

/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */
    
    pardiso_chkmatrix  (&mtype, &n, a, offset_A, row_ptr_A, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec(...)                                              */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    // pardiso_printstats (&mtype, &n, a, offset_A, row_ptr_A, &nrhs, b, &error);
    // if (error != 0) {
    //     printf("\nERROR right hand side: %d", error);
    //     exit(1);
    // }
 
/* -------------------------------------------------------------------- */    
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */ 
    phase = 11; 
    // iparm[27] = 1;
    // iparm[23] = 0; // 0 - one-level parallel; 1 - two-level ; defaut = 1
    // iparm[3] = 31;
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, offset_A, row_ptr_A, pardiso_perm, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
    printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
    printf("\nNumber of supernodes = %d", iparm[29]);
   
/* -------------------------------------------------------------------- */    
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */    
    phase = 22;
    int loop = atoi(argv[2]);
    double t1, t2;
    double t = 0;
    // iparm[31] = 1;
    // printf("iparm[23] = %d\n", iparm[23]);
    
    for ( i = 0; i < loop; i++ )
    {
        t1 = microtime();
        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, offset_A, row_ptr_A, pardiso_perm, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
        t2 = microtime() - t1;
        printf("\nFact time in pardiso is: %lf", t2);
        // printf("\niparm[13] = %d iparm[19] = %d", iparm[13], iparm[19]);
        t += t2;
    }
    printf("\nAverage time is: %lf\n", t/loop);

    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
    printf("\nFactorization completed ...\n ");
    printf("\nNumber of supernodes = %d", iparm[29]);

/* -------------------------------------------------------------------- */    
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */    
    // phase = 33;

    // iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
   
    // pardiso (pt, &maxfct, &mnum, &mtype, &phase,
    //          &n, a, offset_A, row_ptr_A, &idum, &nrhs,
    //          iparm, &msglvl, b, x, &error,  dparm);
   
    // if (error != 0) {
    //     printf("\nERROR during solution: %d", error);
    //     exit(3);
    // }

    // printf("\nSolve completed ... ");
    // printf("\nThe solution of the system is: ");
    // for (i = 0; i < n; i++) {
    //     printf("\n x [%d] = % f", i, x[i] );
    // }
    // printf ("\n");

/* -------------------------------------------------------------------- */
/* ..  Back substitution with tranposed matrix A^t x=b                  */
/* -------------------------------------------------------------------- */

    phase = 33;

    iparm[7]  = 1;       /* Max numbers of iterative refinement steps. */
    iparm[11] = 1;       /* Solving with transpose matrix. */
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, offset_A, row_ptr_A, pardiso_perm, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }

    printf("\nSolve completed ... ");
    // printf("\nThe solution of the system is: ");
    // for (i = 0; i < n; i++) {
    //     printf("\n x [%d] = % f", i, x[i] );
    // }
    // printf ("\n");

/* -------------------------------------------------------------------- */    
/* ... compute diagonal elements of the inverse.                        */                                       
/* -------------------------------------------------------------------- */  

    // phase = 33;
    // iparm[11] = 0;       /* Solving with nontranspose matrix. */
    // /* solve for n right hand sides */
    // for (k = 0; k < n; k++) 
    // {
    // 	for (i = 0; i < n; i++) {
    //     	b[i] = 0;
    // 	}
    // 	/* Set k-th right hand side to one. */
    //     b[k] = 1;
  
    // 	pardiso (pt, &maxfct, &mnum, &mtype, &phase,
    //     	     &n, a, offset_A, row_ptr_A, &idum, &nrhs,
    //    		     iparm, &msglvl, b, x, &error,  dparm);
  
	// if (error != 0) {
    //     	printf("\nERROR during solution: %d", error);
    //     	exit(3);
    // 	}

    //     /* save diagonal element */ 
    //     diag[k] = x[k];
    // }*/

/* -------------------------------------------------------------------- */    
/* ... Inverse factorization.                                           */                                       
/* -------------------------------------------------------------------- */  
   
    // if (solver == 0)
    // {
    // 	printf("\nCompute Diagonal Elements of the inverse of A ... \n");
	// phase = -22;
    //     iparm[35]  = 0; /*  overwrite internal factor L */ 

    //     pardiso (pt, &maxfct, &mnum, &mtype, &phase, 
    //              &n, a, offset_A, row_ptr_A, &idum, &nrhs,
    //              iparm, &msglvl, b, x, &error,  dparm);

    //    /* print diagonal elements */
    //    for (k = 0; k < n; k++)
    //    {
    //         int j = offset_A[k]-1;
    //         printf ("Diagonal element of A^{-1} = %32.24e =  %32.24e \n", a[j], diag[k]);
    //    }
    // }   


/* -------------------------------------------------------------------- */    
/* ..  Convert matrix back to 0-based C-notation.                       */
/* -------------------------------------------------------------------- */ 
    for (i = 0; i < n+1; i++) {
        offset_A[i] -= 1;
    }
    for (i = 0; i < nnz; i++) {
        row_ptr_A[i] -= 1;
    }

/* -------------------------------------------------------------------- */    
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */ 
    phase = -1;                 /* Release internal memory. */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, offset_A, row_ptr_A, pardiso_perm, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);

    return 0;
} 
