#include <stdio.h>
#include <unistd.h>
 


void req_pid()
{
	printf("Requesting PID\n");
	getpid();
}
	
void sleeping()
{
	printf("Sleeping...\n");
	sleep(1);
}
       
int main(int argc, char **argv)
{
	req_pid();
	sleeping();
	printf("End!\n");
	return 0;
}
