#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu.h"
#include <sys/time.h>
#include <stdbool.h>
# include <math.h>

#define ROW_LENGTH 1024

// #define FAIL(code)	((code) < NICS_OK)

static int ReadHeader3(FILE *f, int *m, int *n, int *nnz)
{
	char line[ROW_LENGTH];
	int read;
	line[0] = 0;
	*m = *n = *nnz = 0;

	do 
	{
		if (fgets(line, ROW_LENGTH-1, f) == NULL) return 0;
	} while (line[0] == '%');

#ifdef INT64__
#ifdef _WIN32
	if (sscanf(line, "%I64u %I64u %I64u", m, n, nnz) == 3)
#else
	if (sscanf(line, "%llu %llu %llu", m, n, nnz) == 3)
#endif
#else
	if (sscanf(line, "%u %u %u", m, n, nnz) == 3)
#endif
	{
		return 1;
	}
	else
	{
		do
		{ 
#ifdef INT64__
#ifdef _WIN32
			read = fscanf(f, "%I64u %I64u %I64u", m, n, nnz);
#else
			read = fscanf(f, "%llu %llu %llu", m, n, nnz);
#endif
#else
			read = fscanf(f, "%u %u %u", m, n, nnz);
#endif
			if (read == EOF) return 0;
		} while (read != 3);
	}

	return 1;
}

static int ReadHeader2(FILE *f, int *m, int *n, int *nnz)
{
	char line[ROW_LENGTH];
	int read;
	line[0] = 0;
	*m = *n = *nnz = 0;

	do 
	{
		if (fgets(line, ROW_LENGTH-1, f) == NULL) return 0;
	} while (line[0] == '%');

#ifdef INT64__
#ifdef _WIN32
	if (sscanf(line, "%I64u %I64u", m, nnz) == 2)
#else
	if (sscanf(line, "%llu %llu", m, nnz) == 2)
#endif
#else
	if (sscanf(line, "%u %u", m, nnz) == 2)
#endif
	{
		*n = *m;
		return 1;
	}
	else
	{
		do
		{ 
#ifdef INT64__
#ifdef _WIN32
			read = fscanf(f, "%I64u %I64u", m, nnz);
#else
			read = fscanf(f, "%llu %llu", m, nnz);
#endif
#else
			read = fscanf(f, "%u %u", m, nnz);
#endif
			if (read == EOF) return 0;
		} while (read != 2);
	}
	*n = *m;

	return 1;
}

int ReadTripletColumnToSparse(char *file, int *n, int *nnz, \
									 double **ax, int **ai, int **ap)
{
	FILE *fp;
	int err;
	int m, *aj, i, j;
	int pre, cur, num;
	int cnt;

	if (NULL == file || NULL == n || NULL == nnz \
		|| NULL == ax || NULL == ai || NULL == ap) return 0;
	if (*ax != NULL)
	{
		free(*ax);
		*ax = NULL;
	}
	if (*ai != NULL)
	{
		free(*ai);
		*ai = NULL;
	}
	if (*ap != NULL)
	{
		free(*ap);
		*ap = NULL;
	}
	*n = *nnz = 0;

	fp = fopen(file, "r");
	if (NULL == fp) return 0;

	err = ReadHeader3(fp, n, &m, nnz);
	if (!err)
	{
		fclose(fp);
		return err;
	}
	if (m != *n)
	{
		fclose(fp);
		return 0;
	}

	aj = (int *)malloc(sizeof(int)*(*nnz));
	*ax = (double *)malloc(sizeof(double)*(*nnz));
	*ai = (int *)malloc(sizeof(int)*(*nnz));
	*ap = (int *)malloc(sizeof(int)*(m+1));
	if (NULL == aj || NULL == *ax || NULL == *ai || NULL == *ap)
	{
		fclose(fp);
		if (aj != NULL) free(aj);
		if (*ax != NULL)
		{
			*ax = NULL;
			free(*ax);
		}
		if (*ai != NULL)
		{
			*ai = NULL;
			free(*ai);
		}
		if (*ap != NULL)
		{
			*ap = NULL;
			free(*ap);
		}
		return 0;
	}

	for (i=0; i<*nnz; ++i)
	{
#ifdef INT64__
#ifdef _WIN32
		cnt = fscanf(fp, "%I64u %I64u %lf", &((*ai)[i]), &aj[i], &((*ax)[i]));
#else
		cnt = fscanf(fp, "%llu %llu %lf", &((*ai)[i]), &aj[i], &((*ax)[i]));
#endif
#else
		cnt = fscanf(fp, "%u %u %lf", &((*ai)[i]), &aj[i], &((*ax)[i]));
#endif
		if (cnt != 3)
		{
			free(aj);
			fclose(fp);
			return 0;
		}

		--((*ai)[i]);
		--aj[i];

		if ((*ai)[i] >= *n || aj[i] >= *n)
		{
			free(aj);
			fclose(fp);
			return 0;
		}
	}
	fclose(fp);

	(*ap)[0] = 0;
	pre = 0;
	cur = 0;
	num = 0;

	for (i=0; i<*nnz; ++i)
	{
		cur = aj[i];
		if (pre == cur)
		{
			++num;
		}
		else
		{
			num += (*ap)[pre];
			for (j=pre+1; j<=cur; j++)
			{
				(*ap)[j] = num;
			}
			pre = cur;
			num = 1;
		}
	}
	num += (*ap)[cur];
	for (i=cur+1; i<=m; i++)
	{
		(*ap)[i] = num;
	}

	free(aj);
	return 1;
}

int ReadTripletRowToSparse(char *file, int *n, int *nnz, \
									 double **ax, int **ai, int **ap)
{
	FILE *fp;
	int err;
	int m, *aj, i, j;
	int pre, cur, num;
	int cnt;

	if (NULL == file || NULL == n || NULL == nnz \
		|| NULL == ax || NULL == ai || NULL == ap) return 0;
	if (*ax != NULL)
	{
		free(*ax);
		*ax = NULL;
	}
	if (*ai != NULL)
	{
		free(*ai);
		*ai = NULL;
	}
	if (*ap != NULL)
	{
		free(*ap);
		*ap = NULL;
	}
	*n = *nnz = 0;

	fp = fopen(file, "r");
	if (NULL == fp) return 0;

	err = ReadHeader3(fp, n, &m, nnz);
	if (!err)
	{
		fclose(fp);
		return err;
	}
	if (m != *n)
	{
		fclose(fp);
		return 0;
	}

    printf("func!\n");
	aj = (int *)malloc(sizeof(int)*(*nnz));
	*ax = (double *)malloc(sizeof(double)*(*nnz));
	*ai = (int *)malloc(sizeof(int)*(*nnz));
	*ap = (int *)malloc(sizeof(int)*(m+1));
	if (NULL == aj || NULL == *ax || NULL == *ai || NULL == *ap)
	{
		fclose(fp);
		if (aj != NULL) free(aj);
		if (*ax != NULL)
		{
			*ax = NULL;
			free(*ax);
		}
		if (*ai != NULL)
		{
			*ai = NULL;
			free(*ai);
		}
		if (*ap != NULL)
		{
			*ap = NULL;
			free(*ap);
		}
		return 0;
	}

	for (i=0; i<*nnz; ++i)
	{
#ifdef INT64__
#ifdef _WIN32
		cnt = fscanf(fp, "%I64u %I64u %lf", &aj[i], &((*ai)[i]), &((*ax)[i]));
#else
		cnt = fscanf(fp, "%llu %llu %lf", &aj[i], &((*ai)[i]), &((*ax)[i]));
#endif
#else
		cnt = fscanf(fp, "%u %u %lf", &aj[i], &((*ai)[i]), &((*ax)[i]));
#endif
		if (cnt != 3)
		{
			free(aj);
			fclose(fp);
			return 0;
		}

		--((*ai)[i]);
		--aj[i];

		if ((*ai)[i] >= *n || aj[i] >= *n)
		{
			free(aj);
			fclose(fp);
			return 0;
		}
	}
	fclose(fp);

	(*ap)[0] = 0;
	pre = 0;
	cur = 0;
	num = 0;

	for (i=0; i<*nnz; ++i)
	{
		cur = aj[i];
		if (pre == cur)
		{
			++num;
		}
		else
		{
			num += (*ap)[pre];
			for (j=pre+1; j<=cur; j++)
			{
				(*ap)[j] = num;
			}
			pre = cur;
			num = 1;
		}
	}
	num += (*ap)[cur];
	for (i=cur+1; i<=m; i++)
	{
		(*ap)[i] = num;
	}

	free(aj);
	return 1;
}

