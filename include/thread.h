/*multi-threaded programming interfaces for both windows and linux*/
/*last modified: june 14, 2013*/
/*author: Chen, Xiaoming*/
/*system requirements: visual studio 2005 or higher for windows and pthread interfaces for linux*/

#ifndef __THREAD__
#define __THREAD__

#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#define __USE_GNU
#include <pthread.h>
#include <semaphore.h>

#define THREAD_DECL				void *
#define THREAD_DECL_TYPE		void *
#define THREAD_RETURN			NULL
typedef void *					(*thread_proc__t)(void *);
typedef pthread_t				thread_id__t;
typedef pthread_mutex_t			lock__t;
typedef sem_t					sem__t;
typedef sem_t					event__t;
typedef pthread_mutex_t			critical_section__t;
#ifndef NO_ATOMIC
typedef volatile int			spin_lock__t;
#endif

#endif

typedef lock__t					mutex__t;

/*interfaces*/
#ifdef __cplusplus
extern "C" {
#endif

/*if ok, return 0, otherwise fail*/

/*thread*/
int				_CreateThread(thread_proc__t proc, void *arg, thread_id__t *id);
int				_WaitThreadExit(thread_id__t id);
thread_id__t	_GetCurrentThread();
int				_BindThreadToCores(thread_id__t id, unsigned int *cores, int ct);
int				_UnbindThreadFromCores(thread_id__t id);

/*lock, i.e. mutex*/
int		_CreateLock(lock__t *lock);
int		_Lock(lock__t *lock);
int		_Unlock(lock__t *lock);
int		_DestroyLock(lock__t *lock);
#define _CreateMutex	_CreateLock
#define _MutexLock		_Lock
#define _MutexUnlock	_Unlock
#define _DestroyMutex	_DestroyLock

/*semaphore*/
int		_CreateSemaphore(sem__t *sem, int initval);
int		_WaitSemaphore(sem__t *sem);
int		_IncreaseSemaphore(sem__t *sem);
int		_DestroySemaphore(sem__t *sem);

/*event*/
int		_CreateEvent(event__t *ev, int initval);
int		_CreateEventA(event__t *ev, int initval);
int		_WaitEvent(event__t *ev);
int		_WaitEventA(event__t *ev);
int		_SetEvent(event__t *ev);
int		_ResetEvent(event__t *ev);
int		_DestroyEvent(event__t *ev);

/*critical section*/
int		_InitializeCriticalSection(critical_section__t *cs);
int		_EnterCriticalSection(critical_section__t *cs);
int		_LeaveCriticalSection(critical_section__t *cs);
int		_DeleteCriticalSection(critical_section__t *cs);

#ifndef NO_ATOMIC
/*spin*/
int		_SpinInit(spin_lock__t *spin);
int		_SpinLock(spin_lock__t *spin);
int		_SpinUnlock(spin_lock__t *spin);
#endif

/*delay*/
void	_Delay(unsigned int ms);


/*spin wait, for simple event wait*/
#ifdef _WIN32
void	_SpinWaitInt64(volatile __int64 *);
#else
void	_SpinWaitInt64(volatile long long *);
#endif
void	_SpinWaitInt(volatile int *);
void	_SpinWaitShort(volatile short *);
void	_SpinWaitChar(volatile char *);
void	_SpinWaitFloat(volatile float *);
void	_SpinWaitDouble(volatile double *);
void	_SpinWaitSizeInt(volatile size_t *);
void	_SpinBarrier(int, int, volatile char *);

#ifdef __cplusplus
}
#endif

