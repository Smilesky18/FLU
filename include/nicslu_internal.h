#ifndef __NICSLU_INTERNAL__
#define __NICSLU_INTERNAL__

#include "nicslu.h"
#include "thread.h"

#define OK(code)						((code) >= NICS_OK)
#define FAIL(code)						((code) < NICS_OK)
#define WARNING(code)					((code) > NICS_OK)

/*warning code*/
#define NICSLU_MATRIX_NOT_SORTED		(1)
#define NICSLU_WORK_EXIT				(-1)
#define NICSLU_WORK_NONE				(0)
#define NICSLU_WORK_FACT_CLUSTER		(1)
#define NICSLU_WORK_FACT_PIPELINE		(2)
#define NICSLU_WORK_REFACT_CLUSTER		(3)
#define NICSLU_WORK_REFACT_PIPELINE		(4)
#define NICSLU_WORK_COPY_DATA			(5)


#endif
