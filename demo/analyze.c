#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
# include <math.h>

int FLU_Analyze(int n, int *ap, int *ai, int order)
{
    double amd_Info [20];
    int result;

    result = AMD_order (n, ap, ai, Pblk, NULL, amd_Info) ;
}
