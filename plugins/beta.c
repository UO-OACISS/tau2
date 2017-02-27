#include <stdlib.h>
#include <stdio.h>

int plugin_init_beta()
{
 fprintf(stdout, "Beta plugin init ..\n");
 return 1;
}
