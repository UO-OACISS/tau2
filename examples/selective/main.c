#include <stdio.h>

void F1(void);
int other_c_routine(void)
{
	printf("Inside other_c_routine\n");
	return 0;
}
int main(int argc, char **argv)
{
	/* Invoke program with --profile Fort1+Fort2  */
	int i;
	
	/* Comment this out */
#if (defined (PROFILING_ON) || defined (TRACING_ON))
	 TAU_DISABLE_GROUP_NAME("Fort2");
#endif  /* PROFILING_ON */

	for (i = 0; i < argc; i++)
	{ 
		printf("Argv[%d] = %s\n", i, argv[i]);
	}
	 F1();
	 printf("Inside main: after calling f1()\n");
	 other_c_routine();
	 return 0;
}
