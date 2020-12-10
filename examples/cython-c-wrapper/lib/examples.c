#include <stdio.h>

#include "examples.h"

#ifdef __cplusplus
extern "C"
#endif
void hello_from_c_function(const char *name) {
    printf("hello %s\n", name);
}