#ifndef _TAU_SAMPLING_UNWIND_H
#define _TAU_SAMPLING_UNWIND_H

/* *CWL* - The only purpose of this file is to enforce the definition of
           common functionality to be implemented by exactly one unwinder.
	   Failure results in TAU build-time errors rather than 
	   silent "success". Works ONLY because they are not extern "C"
*/
#include <Profile/Profiler.h>
#include <vector>
// Putting "using namespace" statements in header files can create ambiguity
// between user-defined symbols and std symbols, creating unparsable code
// or even changing the behavior of user codes.  This is also widely considered
// to be bad practice.  Here's a code PDT can't parse because of this line:
//   EX: #include <complex>
//   EX: typedef double real;
//
//using namespace std;



void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
				       void *context);
void Tau_sampling_unwindTauContext(int tid, void **address);
std::vector<unsigned long> *Tau_sampling_unwind(int tid, Profiler *profiler,
					   void *pc, void *context);

/* *CWL* - Looks like we do need to add common unwinding prototypes here as well. 
   These will be declared and implemented in TauSampling.cpp.
 */
extern "C" bool unwind_cutoff(void **addresses, void *address);
#endif /* _TAU_SAMPLING_UNWIND_H */
