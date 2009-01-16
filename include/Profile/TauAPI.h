/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.h					  **
**	Description 	: TAU Profiling Package API			  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _TAU_API_H_
#define _TAU_API_H_

#if (defined(PROFILING_ON) || defined(TRACING_ON) )

extern "C" void Tau_start(const char *name);
extern "C" void Tau_stop(const char *name);
extern "C" void Tau_start_timer(void * function_info, int phase );
extern "C" int Tau_stop_timer(void * function_info); 
extern "C" int Tau_stop_current_timer();
extern "C" void Tau_create_top_level_timer_if_necessary(void);
extern "C" void Tau_stop_top_level_timer_if_necessary(void);
extern "C" char * Tau_phase_enable(const char *group);

extern "C" void Tau_profile_snapshot(char *name);
extern "C" void Tau_metadata(char *name, char *value);
extern "C" void Tau_phase_metadata(char *name, char *value);
extern "C" void Tau_context_metadata(char *name, char *value);
extern "C" void Tau_profile_snapshot_1l(char *name, int number);
extern "C" void Tau_dynamic_start(char *name, int isPhase); 
extern "C" void Tau_dynamic_stop(char *name, int isPhase); 
extern "C" void Tau_static_phase_start(char *name);
extern "C" void Tau_static_phase_stop(char *name);
extern "C" void* Tau_get_profiler(const char *name, const char *type, TauGroup_t group, const char *gr_name);
extern "C" void Tau_get_calls(void *handle, long* values, int tid);
extern "C" void Tau_get_child_calls(void *handle, long* values, int tid);
extern "C" void Tau_get_inclusive_values(void *handle, double* values, int tid);
extern "C" void Tau_get_exclusive_values(void *handle, double* values, int tid);
extern "C" void Tau_get_counter_info(const char ***counterlist, int *numcounters);
extern "C" int  Tau_get_tid(void);
extern "C" void Tau_destructor_trigger();


extern "C" void Tau_profile_set_name(void *ptr, const char *name);
extern "C" void Tau_profile_set_type(void *ptr, const char *type);
extern "C" void Tau_profile_set_group(void *ptr, TauGroup_t group);
extern "C" void Tau_profile_set_group_name(void *ptr, const char *groupname);


extern "C" const char *Tau_profile_get_group_name(void *ptr);
extern "C" const char *Tau_profile_get_name(void *ptr);
extern "C" const char *Tau_profile_get_type(void *ptr);
extern "C" TauGroup_t Tau_profile_get_group(void *ptr);

class FunctionInfo;
class Tau_Profile_Wrapper {
public:
  void *fi;

  inline Tau_Profile_Wrapper(void *fi, int phase = 0) {
    this->fi = fi;
#ifndef TAU_PROFILEPHASE
    phase = 0;
#endif
    if (fi != 0) {
      Tau_start_timer(fi, phase);
    }
  }

  inline ~Tau_Profile_Wrapper() {
    if (fi != 0) {
      Tau_stop_timer(fi);
    }
  }
};


#define TAU_TYPE_STRING(profileString, str) static string profileString(str);


#define TAU_PROFILE(name, type, group) \
	static void *tauFI = 0; \
        if (tauFI == 0) tauCreateFI(&tauFI, name, type, (TauGroup_t) group, #group); \
	Tau_Profile_Wrapper tauFProf(tauFI);


#define TAU_PHASE(name, type, group) \
	static void *tauFInfo = NULL; \
	static char *TauGroupNameUsed = Tau_phase_enable(#group); \
        tauCreateFI(&tauFInfo, name, type, (TauGroup_t) group, TauGroupNameUsed); \
	Tau_Profile_Wrapper tauFProf(tauFInfo, 1);

#define TAU_DYNAMIC_PROFILE(name, type, group) \
	static TauGroup_t tau_dy_group = group; \
        static int tau_timer_counter = 0; \
	void *tauFInfo = NULL; \
        char tau_timer_iteration_number[128]; \
        sprintf(tau_timer_iteration_number, " [%d]", ++tau_timer_counter); \
        tauCreateFI(&tauFInfo, string(name)+string(tau_timer_iteration_number), type, tau_dy_group, #group); \
	Tau_Profile_Wrapper tauFProf(tauFInfo);

#define TAU_DYNAMIC_PHASE(name, type, group) \
	static TauGroup_t tau_dy_group = group; \
        static int tau_timer_counter = 0; \
	void *tauFInfo = NULL; \
        char tau_timer_iteration_number[128]; \
        sprintf(tau_timer_iteration_number, " [%d]", ++tau_timer_counter); \
        tauCreateFI(&tauFInfo, string(name)+string(tau_timer_iteration_number), type, tau_dy_group, #group); \
	Tau_Profile_Wrapper tauFProf(tauFInfo, 1);







/* The macros below refer to TAU's FunctionInfo object. This object is created with
a new call for multi-threaded applications and with a static constructor when 
a single thread of execution is used. Correspondingly we either use tauFI.method() 
or tauFI->method();
*/
#define TAU_PROFILE_SET_GROUP_NAME(newname) Tau_profile_set_group_name(tauFI,newname);
#define TAU_PROFILE_TIMER_SET_NAME(t, newname)	Tau_profile_set_name(t,newname);
#define TAU_PROFILE_TIMER_SET_TYPE(t, newname)  Tau_profile_set_type(t,newname);
#define TAU_PROFILE_TIMER_SET_GROUP(t, id) Tau_profile_set_group(t,id); 
#define TAU_PROFILE_TIMER_SET_GROUP_NAME(t, newname) Tau_profile_set_group_name(t,newname);

#define TAU_PROFILE_TIMER_GET_NAME(timer) Tau_profile_get_name(timer)
#define TAU_PROFILE_TIMER_GET_TYPE(timer) Tau_profile_get_type(timer)
#define TAU_PROFILE_TIMER_GET_GROUP(timer) Tau_profile_get_group(timer)
#define TAU_PROFILE_TIMER_GET_GROUP_NAME(timer) Tau_profile_get_group_name(timer)


/**************************************************************************/



#define TAU_NEW(expr, size) 			Tau_new(__FILE__, __LINE__, size, expr)
#define TAU_DELETE(expr, variable) 		Tau_track_memory_deallocation(__FILE__, __LINE__, variable) , expr




#ifdef NO_RTTI
#define CT(obj) string(" ")
#else // RTTI is present
#define CT(obj) RtsLayer::GetRTTI(typeid(obj).name())
#endif //NO_RTTI


#endif /* PROFILING_ON */

#ifdef DEBUG_PROF
#define DEBUGPROFMSG(msg) { cout<< msg; }
#else
#define DEBUGPROFMSG(msg) 
#endif // DEBUG_PROF

#endif /* _TAU_API_H_ */
/***************************************************************************
 * $RCSfile: TauAPI.h,v $   $Author: amorris $
 * $Revision: 1.85 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: TauAPI.h,v 1.85 2009/01/16 00:46:32 amorris Exp $ 
 ***************************************************************************/
