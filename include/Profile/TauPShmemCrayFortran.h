#ifndef _TAU_PSHMEM_CRAY_FORTRAN_H_
#define _TAU_PSHMEM_CRAY_FORTRAN_H_
#include <shmem.h>

void pstart_pes_(int* npes);
void pshmem_broadcast4_(void* target, void* source, int* nlong, int* peroot, int* pestart, int* log_pestride, int* pesize, long* psync);
void pshmem_get4_(void* target, void* source, int* len, int* pe);
void pshmem_int4_and_to_all_(void* target, void* source, int* nreduce, int* PE_start,int* PE_stride, int* PE_size, int* pWrk, long* pSync);
int  pshmem_int4_cswap_(void* target, int* cond, int* value, int* pe);
int  pshmem_int4_fadd_(void* target, int* value, int* pe);
int  pshmem_int4_finc_(void* target, int* pe);
void pshmem_int4_max_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, int* pwrk, long* psync);
void pshmem_int4_min_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, int* pwrk, long* psync);
void pshmem_int4_or_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, int* pwrk, long* psync);
void pshmem_int4_prod_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, int* pwrk, long* psync);
void pshmem_int4_sum_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, int* pwrk, long* psync);
int  pshmem_int4_swap_(void* target, int* value, int* pe);
void pshmem_int4_wait_(int* var, int* value);
void pshmem_int4_wait_until_(int* var, int* cond, int* value);
void pshmem_int4_xor_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, int* pwrk, long* psync);
void pshmem_put4_(void* target, void* source, int* len, int* pe);
void pshmem_put4_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);
void pshmem_real4_max_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real4_min_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real4_prod_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real4_sum_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real4_swap_(void* target, void* value, int* pe);
void pshmem_broadcast8_(void* target, void* source, long* nlong, int* peroot, int* pestart, int* log_pestride, int* pesize, long* psync);
void pshmem_get8_(void* target, void* source, long* len, int* pe);
void pshmem_int8_and_to_all_(void* target, void* source, int* nreduce, int* pestart,int* pestride, int* pesize, long* pwrk, long* psync);
long pshmem_int8_cswap_(void* target, long* cond, long* value, int* pe);
long pshmem_int8_fadd_(void* target, long* value, int* pe);
long pshmem_int8_finc_(void* target,  int* pe);
void pshmem_int8_max_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, long* pwrk, long* psync); 
void pshmem_int8_min_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, long* pwrk, long* psync);
void pshmem_int8_or_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, long* pwrk, long* psync);
void pshmem_int8_prod_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, long* pwrk, long* psync);
void pshmem_int8_sum_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, long* pwrk, long* psync);
long pshmem_int8_swap_(void* target, long* value, int* pe);
void pshmem_int8_wait_(long* var,long* value);
void pshmem_int8_wait_until_(long* var, int* cond, long* value);
void pshmem_int8_xor_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, long* pwrk, long* psync);
void pshmem_put8_(void* target, void* source, long* len, int* pe);
void pshmem_put8_nb_(void* target, void* source, long* len, int* pe, void* transfer_handle);

void pshmem_real8_max_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real8_min_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real8_prod_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_real8_sum_to_all_(void* target, void* source, int* nreduce,  int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
double pshmem_real8_swap_(void* target, void* value, int* pe);

void pshmem_barrier_(int* pestart, int* log_pestride, int* pesize, long* psync);
void pshmem_barrier_all_(void);
void pshmem_broadcast32_(void* target, void* source, int* nlong, int* peroot, int* pestart, int* log_pestride, int* pesize, long* psync);
void pshmem_broadcast64_(void* target, void* source, int* nlong, int* peroot, int* pestart, int* log_pestride, int* pesize, long* psync);
void pshmem_broadcast_(void* target, void* source, int* nlong, int* peroot, int* pestart, int* log_pestride, int* pesize, long* psync);
void pshmem_character_get_(void* target, void* source, int* len, int* pe);
void pshmem_character_put_(void* target, void* source, int* len, int* pe);
void pshmem_clear_event_(long* event);
void pshmem_clear_lock_(long* lock);
void pshmem_complex_get_(void* target, void* source, int* len, int* pe);
void pshmem_complex_put_(void* target, void* source, int* len, int* pe);
void pshmem_complex_put_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);

void pshmem_double_get_(void* target, void* source, int* len, int* pe);
void pshmem_double_put_(void* target, void* source, int* len, int* pe);
void pshmem_double_put_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);
void pshmem_fence_(void);

void pshmem_finalize_(void);
void pshmem_get128_(void* target, void* source, int* len, int* pe);
void pshmem_get16_(void* target, void* source, int* len, int* pe);
void pshmem_get32_(void* target, void* source, int* len, int* pe);
void pshmem_get64_(void* target, void* source, int* len, int* pe);
void pshmem_get_(void* target, void* source, int* len, int* pe);
void pshmem_getmem_(void* target, void* source, int* len, int* pe);
void pshmem_iget128_(void* target, void* source, void* tst, int* sst, int* len, int* pe);
void pshmem_iget16_(void* target, void* source, void* tst, int* sst, int* len, int* pe);
void pshmem_iget32_(void* target, void* source, void* tst, int* sst, int* len, int* pe);
void pshmem_iget64_(void* target, void* source, void* tst, int* sst, int* len, int* pe);

void pshmem_init_(void);

void pshmem_int2_and_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_int2_max_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_int2_min_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_int2_or_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_int2_prod_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_int2_sum_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);
void pshmem_int2_xor_to_all_(void* target, void* source, int* nreduce, int* pestart, int* log_pestride, int* pesize, void* pwrk, long* psync);

void pshmem_integer_get_(void* target, void* source, int* len, int* pe);
void pshmem_integer_put_(void* target, void* source, int* len, int* pe);
void pshmem_integer_put_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);

void pshmem_iput128_(void* target, void* source, int* tst, int* sst, int* len, int* pe);
void pshmem_iput16_(void* target, void* source, int* tst, int* sst, int* len, int* pe);
void pshmem_iput32_(void* target, void* source, int* tst, int* sst, int* len, int* pe);
void pshmem_iput64_(void* target, void* source, int* tst, int* sst, int* len, int* pe);


void pshmem_logical_get_(void* target, void* source, int* len, int* pe);
void pshmem_logical_put_(void* target, void* source, int* len, int* pe);
void pshmem_logical_put_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);

void start_pes_(int *npes);
int pshmem_my_pe_(void);
int pshmem_n_pes_(void);

void pshmem_put128_(void* target, void* source, int* len, int* pe);
void pshmem_put128_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);
void pshmem_put16_(void* target, void* source, int* len, int* pe);
void pshmem_put16_nb_(void* target, void* source, int* len, int* pe,void* transfer_handle);
void pshmem_put32_(void* target, void* source, int* len, int* pe);
void pshmem_put32_nb_(void* target, void* source, int* len, int* pe,void* transfer_handle);

void pshmem_put64_(void* target, void* source, int* len, int* pe);
void pshmem_put64_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);

void pshmem_put_(void* target, void* source, int* len, int* pe);
void pshmem_put_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);
void pshmem_putmem_(void* target, void* source, int* len, int* pe);
void pshmem_putmem_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);
void pshmem_quiet_(void);

void pshmem_real_get_(void* target, void* source, int* len, int* pe);
void pshmem_real_put_(void* target, void* source, int* len, int* pe);
void pshmem_real_put_nb_(void* target, void* source, int* len, int* pe, void* transfer_handle);
void pshmem_set_event_(long *event);
void pshmem_set_lock_(long *lock);

int pshmem_test_event_(long *event);
int pshmem_test_lock_(long *lock);

void pshmem_wait_(int* var, void* value);
void pshmem_wait_event_(long *event);
void pshmem_wait_until_(void* var, int* cond, long* value);
#endif /*  _TAU_PSHMEM_CRAY_FORTRAN_H_ */
