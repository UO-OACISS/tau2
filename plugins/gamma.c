#include <stdlib.h>
#include <stdio.h>

int plugin_init_gamma()
{
 fprintf(stdout, "Gamma plugin init ..\n");
 return 1;
}
