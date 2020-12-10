#include <stdio.h>
#include <omp.h>

#include "examples.h"

void hello_from_thread(const char* name) {
    printf("hello %s from thread %d\n", name, omp_get_thread_num());
}

#ifdef __cplusplus
extern "C"
#endif
void hello_from_c_function(const char *name) {
#pragma omp parallel
    hello_from_thread(name);
}