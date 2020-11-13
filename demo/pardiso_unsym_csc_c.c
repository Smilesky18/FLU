/*******************************************************************************
* Copyright 2004-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*
*   Content : Intel(R) MKL PARDISO C example program to show the use of the "PARDISO"
*              routine on unsymmetric linear systems in CSC format.
*
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"
#include "nicslu.h"
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00

/* Time Stamp */
double microtime()
{
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

MKL_INT main (int argc[], char *argv[])
{
    /* Matrix data in CSC format. */
    /*MKL_INT n = 5;
    MKL_INT ia[6] = { 1, 4, 7, 9, 12, 14};
    MKL_INT ja[13] = 
    { 1, 2,    4,
      1, 2,       5,
            3, 4,
      1,    3, 4,
            3,    5
    };
    double a[13] = 
    { 1.0,-2.0,     -4.0,
     -1.0, 5.0,           8.0,
                4.0, 2.0,
               -3.0, 6.0, 7.0,
      4.0,     -5.0
    };*/
    int m1, n1, nnz1, ii;

    ReadMatrixMarketFile(argv[1], &m1, &n1, &nnz1, NULL, NULL, NULL, NULL, NULL, NULL);

    // double a[nnz1];
    // int ia1[n1+1];
    // int ja1[nnz1];

    double *a = (double *)malloc(sizeof(double)*nnz1);
    int *ja1 = (int *)malloc(sizeof(int)*nnz1);
    int *ia1 = (int *)malloc(sizeof(int)*(1 + n1));

    ReadMatrixMarketFile(argv[1], &m1, &n1, &nnz1, a, ja1, ia1, NULL, NULL, NULL); // CSC Read
    printf("***********%s: row %d, col %d, nnz %d\n", argv[1], n1, n1, nnz1);
	// SparseTranspose(n1, a, ja1, ia1, 0); //Matrix transpose, CSR Read
    // printf("After transpose!\n");

    for ( ii = 0; ii < nnz1; ii++ ) ja1[ii]+=1;
    for ( ii = 0; ii < n1+1; ii++ ) ia1[ii]+=1;

    // for ( int ii = 0; ii < n1+1; ii++ ) 
    // {
    //     printf("ia1[%d] = %d\n", ii, ia1[ii]);
    // }

    MKL_INT n = (MKL_INT) n1;
    MKL_INT *ia = (MKL_INT *)malloc(sizeof(MKL_INT)*(1 + n));
    MKL_INT *ja = (MKL_INT *)malloc(sizeof(MKL_INT)*(nnz1));

    printf("n = %d\n", n);

    for ( int ii = 0; ii < n+1; ii++ ) 
    {
        ia[ii] = (MKL_INT)ia1[ii];
        // printf("ia[%d] = %d\n", ii, ia[ii]);
    }
    for ( int ii = 0; ii < nnz1; ii++ ) 
    {
        ja[ii] = (MKL_INT)ja1[ii];
        // printf("a[%d] = %lf\n", ii, a[ii]);
    }

    /*MKL_INT n = 5;
    MKL_INT ia[6] = { 1, 4, 7, 9, 12, 14};
    MKL_INT ja[13] = 
    { 1, 2,    4,
      1, 2,       5,
            3, 4,
      1,    3, 4,
            3,    5
    };
    double a[13] = 
    { 1.0,-2.0,     -4.0,
     -1.0, 5.0,           8.0,
                4.0, 2.0,
               -3.0, 6.0, 7.0,
      4.0,     -5.0
    };*/

    /*for ( ii = 0; ii < n; ii++ )
    {
        printf("\ncolumn %d: ", ii);
        for ( int j = ia[ii]; j < ia[ii+1]; j++ )
        {
            printf("%d ", ja[j]);
        }
        // printf("ia[%d] = %d\n", ii, ia[ii]);
    }*/

    MKL_INT mtype = 11;       /* Real unsymmetric matrix */
  // Descriptor of main sparse matrix properties
  struct matrix_descr descrA;
  // Structure with sparse matrix stored in CSR format
  sparse_matrix_t       csrA;
    /* RHS and solution vectors. */
    double *b = (double *)malloc(sizeof(double) * n);
    double *x = (double *)malloc(sizeof(double) * n);
    double *bs = (double *)malloc(sizeof(double) * n);
    double res, res0;
    // double b[n], x[n], bs[n], res, res0;
    MKL_INT nrhs = 1;     /* Number of right hand sides. */
    /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    void *pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    MKL_INT i, j;
    double ddum;          /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */
/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
    for ( i = 0; i < 64; i++ )
    {
        iparm[i] = 0;
    }
    iparm[0] = 1;         /* No solver default */
    iparm[1] = 2;         /* Fill-in reordering from METIS */ // 0 - AMD; 2 - METIS
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 2;         /* Max numbers of iterative refinement steps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Conjugate/transpose solve */
    iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;         /* Which factorization to use. */
    msglvl = 0;           /* Print statistical information  */
    error = 0;            /* Initialize error flag */
/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* -------------------------------------------------------------------- */
    for ( i = 0; i < 64; i++ )
    {
        pt[i] = 0;
    }
/* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates */
/* all memory that is necessary for the factorization. */
/* -------------------------------------------------------------------- */
    phase = 11;
    // printf("Before PARDISO n = %d\n", n);
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during symbolic factorization: %d", error);
        exit (1);
    }
    printf ("\nReordering completed ... ");
    printf ("\nNumber of nonzeros in factors = %d", iparm[17]);
    printf ("\nNumber of factorization MFLOPS = %d\n", iparm[18]);
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */
    phase = 22;
    double t1, t2;
    double t = 0;
    int loop = atoi(argv[2]);
    for ( int k = 0; k < loop; k++ )
    {
        t1 = microtime();
        PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
        t2 = microtime() - t1;
        t += t2;
        // printf("FACT Time = %lf\n", t2);
    }
    printf("Average time = %lf\n", t/loop);
    if ( error != 0 )
    {
        printf ("\nERROR during numerical factorization: %d", error);
        exit (2);
    }
    printf ("\nFactorization completed ... ");
/* -------------------------------------------------------------------- */
/* .. Solution phase. */
/* -------------------------------------------------------------------- */
    phase = 33;

  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  descrA.mode = SPARSE_FILL_MODE_UPPER;
  descrA.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE, n, n, ia, ia+1, ja, a );

    /* Set right hand side to one. */
    for ( i = 0; i < n; i++ )
    {
        b[i] = 1;
    }

// Transpose solve is used for systems in CSC format
    iparm[11] = 2;

    printf ("\n\nSolving the system in CSC format...\n");
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during solution: %d", error);
        exit (3);
    }

 
    // Compute residual
    // the CSC format for A is the CSR format for A transposed
      mkl_sparse_d_mv( SPARSE_OPERATION_TRANSPOSE, 1.0, csrA, descrA, x, 0.0, bs);
    res = 0.0;
    res0 = 0.0;
    for ( j = 1; j <= n; j++ )
    {
        res += (bs[j - 1] - b[j - 1]) * (bs[j - 1] - b[j - 1]);
        res0 += b[j - 1] * b[j - 1];
    }
    res = sqrt (res) / sqrt (res0);
    printf ("\nRelative residual = %e", res);
    // Check residual
    if ( res > 1e-10 )
    {
        printf ("Error: residual is too high!\n");
        exit (10 + i);
    }
    mkl_sparse_destroy(csrA);

/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
    phase = -1;           /* Release internal memory. */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error);
    return 0;
}