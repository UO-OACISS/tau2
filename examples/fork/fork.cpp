////////////////////////////////////////////////////////////////////////
//
//  fork.cpp - An example that illustrates how to register a fork in the
//  child process to accurately get performance data from the executing process.
//
////////////////////////////////////////////////////////////////////////

#include "Profile/Profiler.h"     // for TAU

#include<stdio.h>
#include<unistd.h>

int someA(void);
int someB(void);
int someC(void);
int someD(void);

/* experimenting with fork */
int pID;

int main(int argc, char *argv[])
{
    int i;

    TAU_PROFILE("main", "int (int, char **)", TAU_DEFAULT);
    TAU_PROFILE_INIT(argc, argv);
    TAU_PROFILE_SET_NODE(0);
    TAU_PROFILE_SET_CONTEXT(0);

    printf("Inside main\n");
    someA();
    return 0;
}

int someA()
{
	TAU_PROFILE("someA","void (void)", TAU_USER);
	printf("Inside someA - sleeping for 3 secs\n");
	sleep(3);
	someB();
	return(0);
}

int someB()
{
	TAU_PROFILE("someB","void (void)", TAU_USER);
	printf("Inside someB - sleeping for 5 secs\n");
	sleep(5);
	someC();
	return(0);
}

int someC()
{
	TAU_PROFILE("someC","void (void)", TAU_USER);
	printf("Inside someC before fork\n");

	pID = fork();
	if (pID == 0)
	{
		printf("Parent : pid returned %d\n", pID);
	}
	else
	{
	 // If we'd used the TAU_INCLUDE_PARENT_DATA, we'd get the performance 
	 // data from the parent in this process as well. 
		TAU_REGISTER_FORK(pID, TAU_EXCLUDE_PARENT_DATA);
		printf("Child : pid = %d - sleeping for 2 secs\n", pID);
		sleep(2);
		someD();
	}
	return(0);
}

int someD()
{
	TAU_PROFILE("someD","void (void)", TAU_USER);
	printf("Inside someD in the child - sleeping for 9 secs\n");
	sleep(9);
	return(0);
}

/***************************************************************************
 * $RCSfile: fork.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2000/10/11 18:58:50 $
 * VERSION: $Id: fork.cpp,v 1.1 2000/10/11 18:58:50 sameer Exp $
 ***************************************************************************/

