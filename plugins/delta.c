#include <stdlib.h>
#include <stdio.h>

int plugin_init_delta()
{
 fprintf(stdout, "Delta plugin init ..\n");
 return 1;
}
