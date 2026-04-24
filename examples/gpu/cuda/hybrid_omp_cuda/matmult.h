#include <stdlib.h>
#include <iostream>
#include "cuda_runtime_api.h"
void multiply_by_element(dim3 a, dim3 b, float *c, float *d, float *e, int f); 
void multiply_by_block(dim3 a, dim3 b, float *c, float *d, float *e, int f); 
