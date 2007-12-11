#include <stdio.h>

//#include <Profile/Profiler.h>

#include "bsp.h"
#include <sys/time.h>


//default is 5000 microsecs
#define DEF_TIME 5000	

#ifdef BGL_TIMERS
extern "C" {
/* header files for BlueGene/L */
#include <bglpersonality.h>
#include <rts.h>
double bsp_rdtsc() {
   static double bgl_clockspeed = 0.0;
   if (bgl_clockspeed == 0.0)
   {
     BGLPersonality mybgl;
     rts_get_personality(&mybgl, sizeof(BGLPersonality));
     bgl_clockspeed = 1.0e6/(double)BGLPersonality_clockHz(&mybgl);
   }
   return (rts_get_timebase() * bgl_clockspeed);
}

double bsp_get_tsc() { return 1000000; }
}
#elif defined(LINUX_TIMERS)
extern "C" unsigned long long getLinuxHighResolutionTscCounter(void);
double KTauGetMHz(void);
extern "C" {
double bsp_rdtsc() { return ((double) getLinuxHighResolutionTscCounter()/1.0); }
double bsp_get_tsc() { return (KTauGetMHz() * 1000000); }
}
#else //not BGL_TIMERS nor LINUX_TIMERS - so use gtod
extern "C" unsigned long long getLinuxHighResolutionTscCounter(void);
double KTauGetMHz(void);
extern "C" {
double bsp_rdtsc() {
	struct timeval tv;
	gettimeofday(&tv,  NULL);
	return ( (double) (tv.tv_sec*1e6 + tv.tv_usec) / 1.0);
}
double bsp_get_tsc() { return 1000000.0; }
}
#endif //BGL_TIMERS

static volatile unsigned long long ktau_inject_dummy = 0;
//a piece of inline code that busy-loops for 'flag' number of times
static inline void work_now(int lc_flag) {
        volatile unsigned long long a = 59, b = 43;
        while(lc_flag--) {
                a += (ktau_inject_dummy + (b/a) - 32);
        }
        ktau_inject_dummy = b+a;
}

extern "C" long long calc_work_loops(unsigned long long micros) {
	unsigned long long start = 0, stop = 0, min = 0xFFFFFFFFFFFFFFFF;
	double mintime = 0.0;
	long long workloops = 0;
	int i = 75000;
	#define CHECK_NO 1000
	while(i--) {
		start = bsp_rdtsc();
		work_now(CHECK_NO);
		stop = bsp_rdtsc();
		if((stop - start) < min) {
			min = stop - start;
		}
	}
	
	//we know CHECK_NOW of work-iters take min time 
	// - so how many iter for micros?
	//first change min to time
	mintime = (min / bsp_get_tsc())*1000000; //we need mintime in micros
	
	workloops = (micros/mintime)*CHECK_NO;
	
	return workloops;
}

/*
int main(int argc, char* argv[]) {
	unsigned long long micros = DEF_TIME;
	long long workloops = 0;
	int i = 5;
	unsigned long long start = 0, stop = 0;
	double time = 0;
	if(argc > 1) {
		micros = strtoull(argv[1], NULL, 10);
		if(micros <= 0) {
			printf("Bad micros:%llu, setting to default:%llu\n", micros, DEF_TIME);
			micros = DEF_TIME;
		}
	}

	workloops = calc_work_loops(micros);

	printf("workloops:%lld\n", workloops);

	printf("Testing ....");
	i = 5;
	while(i--) {
		start = bsp_rdtsc();
		work_now(workloops);
		stop = bsp_rdtsc();
		time = ((stop - start)/bsp_get_tsc()) * 1000000;
		printf("Took: %lf micros. Error:%lf micros.\n", time, time-micros);
	}
	
	return 0;
}
*/
