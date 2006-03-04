// testTau.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

/*** For regular profiling: ***/
//#define PROFILING_ON 1
//#pragma comment(lib, "tau-profile-mt.lib")

/*** For callpath profiling: ***/
//#define PROFILING_ON 1
//#define TAU_CALLPATH 1
//#pragma comment(lib, "tau-callpath-mt.lib")

/*** For tracing: ***/
//#define TRACING_ON 1
//#pragma comment(lib, "tau-trace.lib")

#define TAU_WINDOWS
#define TAU_DOT_H_LESS_HEADERS


#include <Profile/Profiler.h>
#include <windows.h>
#include <process.h>


void ThreadProc(void *arg) {
	TAU_REGISTER_THREAD();
	TAU_PROFILE("ThreadProc", "void (void*)", TAU_USER1);

	int i = (int) arg;
	printf ("Thread %d here\n",i);

	Sleep(i * 1000);

//	_endthread();
}

int bar(int a, int b, int c) {
	TAU_PROFILE("int bar(int, int, int)", " ", TAU_USER1);
//	return 0;
	int d=0;
	d = a + b * c * a;
	for (int i=1;i<20000000;i++) {
		d = a + b * c * a * b * c + a + b * d * 3.1453452345 + c * a + b * c * b * d * 1000;
	}
	return d;
}

int baz(int a, int b, int c) {
	TAU_PROFILE("int baz(int, int, int)", " ", TAU_USER2);
//	return 0;
	int d=0;
	d = a + b * c * a;
	for (int i=1;i<20000000;i++) {
		d = a + b * c * a * b * c + a + b * d * 3.1453452345 + c * a + b * c * b * d * 1000;
	}
	return d;
}

int foo(int a, int b) {
	TAU_PROFILE("int foo(int, int)", " ", TAU_USER);
	double c=0;
	for (int i=0;i<1;i++) {
		c*=bar(a,b,a*b) + baz(b,a,a*b*b) ;
	}
		
	return c;
}

int _tmain(int argc, _TCHAR* argv[])
{
	TAU_PROFILE_INIT(argc, argv);
	TAU_PROFILE_SET_NODE(0);
	TAU_PROFILE("int main(int, char **)", " ", TAU_DEFAULT);
  
#ifndef TAU_MPI
	TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */

	HANDLE threads[3];
	int i;

	for (i=0;i<3;i++) {
		threads[i] = (HANDLE) _beginthread (ThreadProc, 0, (void*)(i+1));
	}

	foo (2,5);
	baz(3,4,5);

	for (i=0;i<3;i++) {
		WaitForSingleObject(threads[i],INFINITE);
	}

	return 0;
}

