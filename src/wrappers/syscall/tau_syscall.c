#define _GNU_SOURCE

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <TAU.h>
#include <stdlib.h>
#include <inttypes.h>

#include <tracee.h>

extern void Tau_profile_exit_all_threads(void);
extern int Tau_init_initializeTAU(void);

// /proc/pid/task/tid/children

pid_t get_child_pid_of_pid(pid_t pid) {
    intmax_t ret = -1;

    char *buf;
    int r = asprintf(&buf, "/proc/%jd/stat", (intmax_t) pid);
    if (r == -1) 
    {
        return ret;
    }

    FILE *f = fopen(buf, "r");
    if (f == NULL)
    {
        free(buf);
        return ret;
    }

    if (fscanf(f, "%jd", &ret) != 1) 
    {
        fclose(f);
        free(buf);
    }
    
    return (pid_t) ret;
}

int main (int argc, char* argv[]) 
{
    pid_t rpid = fork();

    if (rpid < 0)
    {
        perror("fork()");
        exit(EXIT_FAILURE);
    }

    int ret;
    if (rpid == 0)
    {
        // Child
		char **myargv = malloc(sizeof(char *) * (argc + 1) );

		if(!myargv)
		{
			perror("malloc");
			return EXIT_FAILURE;
		}

		int cnt = 0;

		for(int i = 1; i < argc; i++)
		{
			myargv[cnt] = strdup(argv[i]);
			cnt++;
		}

		myargv[cnt] = NULL;
        prepare_to_be_tracked(getppid());
        ret = execvp(myargv[0], myargv);
    }
    else
    {
        // Parent
        TAU_INIT(&argc, &argv);
        Tau_init_initializeTAU();

        int tmp = TAU_PROFILE_GET_NODE();
        if (tmp == -1)
        {
            TAU_PROFILE_SET_NODE(0);
        }
        TAU_PROFILE_SET_CONTEXT(1);

        pid_t test = -1;
        while (test == -1)
        {
            test = get_child_pid_of_pid(rpid);
            printf("test = %d\n", test);
	    fflush(stdout);
            sleep(1);
        }

        ret = track_process(rpid);
        Tau_profile_exit_all_threads();
        Tau_destructor_trigger();
    }

    return ret;
}

