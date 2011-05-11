
#define TAU_WINDOWS
#define TAU_DOT_H_LESS_HEADERS


#include <Profile/Profiler.h>
#include <windows.h>
#include <process.h>


__declspec(dllexport) void dll_func(int a) {
	TAU_PROFILE_TIMER(timer, "dll_func", "void (int)", TAU_USER1);
	TAU_PROFILE_START(timer);
	//printf ("dll_func here: %d\n",a);
	TAU_PROFILE_STOP(timer);
}
