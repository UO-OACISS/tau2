#ifndef _TAU_SAMPLING_UNWIND_H
#define _TAU_SAMPLING_UNWIND_H

/* *CWL* - The only purpose of this file is to enforce the definition of
           common functionality to be implemented by exactly one unwinder.
	   Failure results in TAU build-time errors rather than 
	   silent "success". Works ONLY because they are not extern "C"
*/
#include <Profile/Profiler.h>
#include <vector>
using namespace std;

void Tau_sampling_outputTraceCallstack(int tid, void *pc, 
				       void *context);
void Tau_sampling_unwindTauContext(int tid, void **address);
vector<unsigned long> *Tau_sampling_unwind(int tid, Profiler *profiler,
					   void *pc, void *context);

/* *CWL* - Looks like we do need to add common unwinding prototypes here as well. 
   These will be declared and implemented in TauSampling.cpp.
 */
extern "C" bool unwind_cutoff(void **addresses, void *address);
#endif /* _TAU_SAMPLING_UNWIND_H */
