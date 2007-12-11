#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include <Profile/Profiler.h>

#include "bsp.h"

#ifdef USE_SELFISH
extern void selfish_detour_init(int *val, int *rank);
extern void selfish_detour_finalize();
#endif //USE_SELFISH

#define TYPE_BARRIER 0
#define TYPE_ALLREDUCE 1
#define TYPE_ALLTOALL 2

#define DEBUG 1

int g_rank = -1;

typedef struct _bsp_params {
	long compute_time; //per-phase comp time, micros
	int strong; //strong or weak scaling?	
	long num_phases; //number of iterations
	unsigned long long tot_secs; //overall total time to run
	int random; //is compte-time across ranks balanced or random
	long percent_rand; //if random, what percent of it?
	int collective_type; //type of collective to use
	long workloops; //to be computed amount of "work" (not time)
	int comm_size; //number of processors
	int selfish; //do we get selfish started?
        long pinned; //if >= 0, then indicates number of processors/node & indicates we want to pin tasks
        long timed; //if >= 0, then indicates we should time each comp-phase - by default its 1.
} bsp_params;

bsp_params g_params = {
	.compute_time = 1000,
	.strong = 0,
	.num_phases = 10000,
	.tot_secs = 60,
	.random = 0,
	.percent_rand = 0,
	.collective_type = TYPE_BARRIER,
	.workloops = 0,
	.comm_size = -1,
	.selfish = 0,
        .pinned = -1,
	.timed = 1,
};

void get_params(bsp_params* params) {
	char* env = NULL;
	if((env = getenv("BSP_COMP_TIME"))) {
		if(!g_rank) printf("get_params: Found BSP_COMP_TIME:%s\n", env);
		params->compute_time = strtol(env, NULL, 10);
	}
	if(!g_rank) printf("get_params: compute_time:%ld micros.\n", params->compute_time);

	if((env = getenv("BSP_STRONG"))) {
		params->strong = 1;
		if(!g_rank) printf("get_params: strong scaling is true.(Gave a large enough compute time?)\n");
	} else {
		if(!g_rank) printf("get_params: weak scaling is true.\n");
	}

	if((env = getenv("BSP_PHASES"))) {
		params->num_phases = strtol(env, NULL, 10);
	}
	if(!g_rank) printf("get_params: num_phases:%ld. \n", params->num_phases);

	if((env = getenv("BSP_TOT_TIME"))) {
		params->tot_secs = strtoull(env, NULL, 10);
	}
	if(!g_rank) printf("get_params: tot_secs:%llu secs.\n", params->tot_secs);

	if((env = getenv("BSP_RANDOM"))) {
		params->random = 1;
		if(!g_rank) printf("get_params: Random workload change is true.\n");
	}

	if((env = getenv("BSP_RANDOM_PERCENT"))) {
		params->percent_rand = strtol(env, NULL, 10);
		if(!g_rank) printf("get_params: percent_rand:%ld.\n", params->percent_rand);
	}

	if((env = getenv("BSP_COLLECTIVE"))) {
		if(!strcmp(env, "BARRIER")) {
			params->collective_type = TYPE_BARRIER;
			if(!g_rank) printf("get_params: collective_type:%s.\n", env);
		} else if(!strcmp(env, "ALLREDUCE")) { 
			params->collective_type = TYPE_ALLREDUCE;
			if(!g_rank) printf("get_params: collective_type:%s.\n", env);
		} else if(!strcmp(env, "ALLTOALL")) { 
			params->collective_type = TYPE_ALLTOALL;
			if(!g_rank) printf("get_params: collective_type:%s.\n", env);
		} else {
			if(!g_rank) printf("get_params: Unknown BSP_COLLECTIVE:%s. Going with Barrier\n", env);
		}
	} else {
		if(!g_rank) printf("get_params: (default) collective_type:BARRIER.\n");
	}

	PMPI_Comm_size(MPI_COMM_WORLD, &(params->comm_size));
	if(!g_rank) printf("get_params: comm_size:%d.\n", params->comm_size);

	if((env = getenv("BSP_SELFISH"))) {
		params->selfish = 1;
	}
	if(!g_rank) printf("get_params: selfish:%d.\n", params->selfish);

        if((env = getenv("BSP_PIN"))) {
                params->pinned = strtol(env, NULL, 10);
        }
        if(!g_rank) printf("get_params: pinned:%ld. \n", params->pinned);

        if((env = getenv("BSP_TIMED_COMP"))) {
                params->timed = strtol(env, NULL, 10);
        }
        if(!g_rank) printf("get_params: timed:%ld. \n", params->timed);
}

static volatile unsigned long long ktau_inject_dummy = 0;
//a piece of inline code that busy-loops for 'flag' number of times
static inline void ktau_inject_now(long lc_flag) {
        volatile unsigned long long a = 59, b = 43;
        while(lc_flag--) {
                a += (ktau_inject_dummy + (b/a) - 32);
        }
        ktau_inject_dummy = b+a;
}

void compute(bsp_params* params) {
	//add more stuff based on strong/weak/random etc
	if(params->strong) {
		ktau_inject_now(params->workloops / params->comm_size);
	} else {
		ktau_inject_now(params->workloops);
	}
}

void collective(bsp_params* params) {
	int send, recv;
	switch(params->collective_type) {
		case TYPE_ALLREDUCE:
		MPI_Allreduce(&send, &recv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		break;

		case TYPE_BARRIER:
		default:
		MPI_Barrier(MPI_COMM_WORLD);
		break;
	}
}

void do_pinning(bsp_params* params) {
        int thep = 0;
        unsigned long mask = 1;
        int no_cpus = 0;

        if(params->pinned <= 0) {
                return;
        }

        no_cpus = params->pinned;

        thep = (g_rank % no_cpus);

        mask = mask << thep;

        printf("do_pinning: no_cpus:%d\tthep:%d\tmask:%lu\tsched_affinity ret:%d\n",no_cpus, thep, mask, sched_setaffinity(getpid(), sizeof(mask), &mask));

        return;
}

long long calc_work_loops(unsigned long long micros);

void usage(char* cmd) {
	printf("Usage: %s:", cmd);
	printf("Env-vars that can be set can be located inside the code in bsp.c in get_params() routine.\nPlease look at that.\n");
}

int main(int argc, char* argv[]) {
	
	long remain_phases = 0;
	int rank = -1, size = -1;
	double start = 0, stop = 0, max = 0, min = 0xFFFFFFFFFFFFFFFF, sum = 0;
	double all_max = 0, all_min = 0, all_avg = 0, avg = 0, all_sum = 0;
	double tsc = bsp_get_tsc();

	double rank0_start = 0, rank0_stop = 0;

	TAU_PROFILE_TIMER(apptimer, "Benchmark", "", TAU_DEFAULT);
	TAU_PROFILE_INIT(argc, argv);


	/* Stupid code fragment - doesnt work
	if(argc != 1 || (!strcmp(argv[1],"--help"))) {
		usage(argv[0]);
		return 0;
	}*/

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	g_rank = rank;

	//get env vars
	get_params(&g_params);

        do_pinning(&g_params);

	if(rank == 0) {
		//figure out how much work to do
		printf("TSC: %lf\n", tsc);
		g_params.workloops = calc_work_loops(g_params.compute_time);
	}	

	//send it out to everybody
	PMPI_Barrier(MPI_COMM_WORLD);
	PMPI_Bcast(&(g_params.workloops), 1, MPI_INT, 0, MPI_COMM_WORLD);

	#ifdef USE_SELFISH
	if(g_params.selfish) {
		selfish_detour_init(NULL, &rank);
	}	
	#endif //USE_SELFISH

	PMPI_Barrier(MPI_COMM_WORLD);
	TAU_PROFILE_START(apptimer);
	PMPI_Barrier(MPI_COMM_WORLD);
		
	//now start off
	rank0_start = bsp_rdtsc();
	remain_phases = g_params.num_phases;
	while(remain_phases--) {
		if(g_params.compute_time) {
			if(g_params.timed) {
				start = bsp_rdtsc();
			}
			//compute
			compute(&g_params);
			//collective
			collective(&g_params);

			if(g_params.timed) {
				stop = (bsp_rdtsc() - start);
				if(stop < min) {
					min = stop;
				} else if(stop > max) {
					max = stop;
				}
				sum += stop;
			}
		} else {
			//only collective
			collective(&g_params);
		}
	}
	
	//end
	PMPI_Barrier(MPI_COMM_WORLD);

	rank0_stop = (bsp_rdtsc() - rank0_start);

	TAU_PROFILE_STOP(apptimer);
	PMPI_Barrier(MPI_COMM_WORLD);

	#ifdef USE_SELFISH
	if(g_params.selfish) {
		selfish_detour_finalize();
	}
	#endif //USE_SELFISH

	//compute avg - so we can share it
	avg = (((double)sum/g_params.num_phases)/tsc)*1000000;

	//calc overall max, min, avgs
	PMPI_Reduce(&min, &all_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	PMPI_Reduce(&max, &all_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	PMPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	all_avg = all_sum/g_params.comm_size;
	
	MPI_Finalize();
	//report time across by root
	if(!g_rank) printf("GLOBAL: MIN:%lf \t MAX:%lf \t SUM:%lf \n", (all_min/tsc)*1000000, (all_max/tsc)*1000000, (all_avg/tsc)*1000000);

	//report time in all ranks
	printf("Rank:%d \t Min:%lf \t Max:%lf \t Avg:%lf\n", rank, ((double)min/tsc)*1000000, ((double)max/tsc)*1000000, avg);

	//print TOTAL TIME
	if(!g_rank) printf("TOTAL MICROSECS: %lf\n", (rank0_stop/tsc)*1000000);

	return 0;
}

