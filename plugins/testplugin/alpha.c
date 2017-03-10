#include <stdlib.h>
#include <stdio.h>

int plugin_init_alpha()
{
 fprintf(stdout, "Alpha plugin init ..\n");
 return 1;
}
