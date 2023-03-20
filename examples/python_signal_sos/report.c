
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

#include "sos.h"
#include "sosa.h"

#define VERBOSE  0
#define log(...) { \
    if (VERBOSE) { \
        printf("== test.c: " __VA_ARGS__); \
        fflush(stdout); \
    }\
}


// Forward decl. of callback for query results:
void MY_feedback_handler(void *sos_context,
        int payload_type, int payload_size, void *payload_data);
// Flag for determining if the results have arrived yet, since the
// query request we send is processed async and delivered when it is ready.
int g_done = 0;
int frame = 0;

int main(int argc, char **argv) {

    MPI_Init(NULL,NULL);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SOS_runtime *sos = NULL;
    //SOS_init(&sos, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
    
    // NOTE: Previous style that activates the callback:
    SOS_init(&sos, SOS_ROLE_CLIENT, SOS_RECEIVES_DIRECT_MESSAGES, MY_feedback_handler);



    g_done = 0;


    int sos_port_env = atoi(getenv("SOS_CMD_PORT"));

    while(1)
    {
        if (VERBOSE) {
            printf("== test.c: Sending query to SOS @ port %d...\n", sos_port_env);
        }

    	int new_frame;
    	do
    	{
    	    SOSA_results *manifest = NULL;
     	    new_frame = 0;
    	    int max_frame_overall = 0;
            char pub_title_filter[2048] = {0};	    
    	    manifest = NULL;
         	SOSA_request_pub_manifest( sos, (SOSA_results **) &manifest, &max_frame_overall,
            pub_title_filter, sos->daemon->remote_host, atoi(sos->daemon->remote_port));
    	    if(max_frame_overall > frame )
    	    {
    		    new_frame = 1;
    	    	//printf("manifest (max frame: %d)\n", max_frame_overall);
                printf("\nNew frame published, requesting data:\n");
    	    }
    	    SOSA_results_destroy(manifest);
    	}while(!new_frame);

	    char my_query[1024];
	    snprintf(my_query, 1024, 
		"SELECT node_id, comm_rank, frame, value_name, value " \
		"FROM viewCombined WHERE (frame == %d ) " \
		"AND value_name like \"\%:PY-\%\"" \
		" ORDER by frame, value_name;", frame);
	    printf("%s\n", my_query);
        SOSA_exec_query(sos, my_query, sos->config.daemon_host, sos_port_env);
	    frame++; 
        while (!g_done) {
       	    usleep(100000);
        }
        g_done = 0;
    }




    if (VERBOSE) {
        printf("SOS test client exiting cleanly!\n");
        fflush(stdout);
    }

    SOS_finalize(sos);
    MPI_Finalize();
    return 0;
}




void
MY_feedback_handler(
        void *sos_context,
        int payload_type,
        int payload_size,
        void *payload_data)
{
    SOSA_results *results = NULL;
    if (VERBOSE) {
        printf("== test.c: Message from SOS received!\n\n");
    }
    
    switch (payload_type) { 

    case SOS_FEEDBACK_TYPE_QUERY:
        SOSA_results_init(sos_context, &results);
        SOSA_results_from_buffer(results, payload_data);
        //if (VERBOSE) {        
	if(results != NULL)
	{
            SOSA_results_output_to(stdout, results, "Query Results", SOSA_OUTPUT_W_HEADER);
        }
        SOSA_results_destroy(results);
        break;

    case SOS_FEEDBACK_TYPE_PAYLOAD:
        // NOTE: The SOS feedback dispatcher will free the buffer when we
        //       return from this function.
        printf("demo_app : Received %d-byte payload --> \"%s\"\n",
                payload_size,
                (unsigned char *) payload_data);
        fflush(stdout);
        break;

    case SOS_FEEDBACK_TYPE_CACHE:
        SOSA_results_init(sos_context, &results);
        SOSA_results_from_buffer(results, payload_data);
        if (VERBOSE) {
            SOSA_results_output_to(stdout, results,
                    "Query Results", SOSA_OUTPUT_W_HEADER);
        }
        SOSA_results_destroy(results);
        break; 
    }
    
    g_done = 1;

    if (VERBOSE) {
        printf("== test.c: Message handler is finished, setting g_done=1 and returning...\n");
    }

    return;
}


