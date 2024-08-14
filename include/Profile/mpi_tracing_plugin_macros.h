#define TAU_DO_TIMER_EXIT ((TauEnv_get_current_timer_exit_params() == 1) && (Tau_time_traced_api_call() == 1))

/* These functions and macros are for creating MPI exit "events" in a plugin trace stream. */
/* They are used by the ADIOS plugin and the SOS plugin. */

#if defined(TAU_SOS) || defined(TAU_ADIOS2)

void Tau_plugin_trace_current_timer(const char * name) {
    /*Invoke plugins only if both plugin path and plugins are specified*/
    if(TauEnv_get_plugins_enabled() && TAU_DO_TIMER_EXIT) {
        Tau_plugin_event_current_timer_exit_data_t plugin_data;
        plugin_data.name_prefix = name;
        Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT, name, &plugin_data);
    }
}

#if defined(TAU_SOS)
#define EVENT_TRACE_PREFIX "TAU_EVENT::MPI"
#else
#define EVENT_TRACE_PREFIX "\"cat\": \"MPI\", \"name\":"
#endif

void convert_comm(char * tmpstr, MPI_Comm comm) {
    if (comm == MPI_COMM_WORLD) {
        sprintf(tmpstr, "MPI_COMM_WORLD");
        return;
    }
    if (comm == MPI_COMM_SELF) {
        sprintf(tmpstr, "MPI_COMM_SELF");
        return;
    }
    if (comm == MPI_COMM_NULL) {
        sprintf(tmpstr, "MPI_COMM_NULL");
        return;
    }
    sprintf(tmpstr, "0x%" PRIx64 "", (uint64_t)comm);
    return;
}

#define TIMER_EXIT_COLLECTIVE_SYNC_EVENT(__desc,__comm) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"%s\", \"comm\": \"%s\"", EVENT_TRACE_PREFIX, __desc, __commstr); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COLLECTIVE_EXCH_EVENT(__desc,__size,__root,__comm) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"%s\", \"size\": %d, \"root\": %u, \"comm\": \"%s\"", EVENT_TRACE_PREFIX, __desc, __size, __root, __commstr); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COLLECTIVE_EXCH_ALL_EVENT(__desc,__send_size,__recv_size,__root,__comm) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"%s\", \"sendsize\": %d, \"recvsize\": %d, \"root\": %u, \"comm\": \"%s\"", EVENT_TRACE_PREFIX, __desc, __send_size, __recv_size, __root, __commstr); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COLLECTIVE_EXCH_V_EVENT(__desc,__label,__mybytes,__stats,__root,__comm) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"%s\", \"%s\": %d, \"count\": %f, \"mean\": %f, \"min\": %f, \"max\": %f, \"sumsqr\": %f, \"root\": %u, \"comm\": \"%s\"", \
    EVENT_TRACE_PREFIX, __desc, __label, __mybytes, __stats[0],__stats[1],__stats[2],__stats[3],__stats[4], __root, __commstr); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COLLECTIVE_EXCH_AAV_EVENT(__desc,__stats1,__stats2,__comm) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  \
    "%s \"%s\", \"sendcount\": %f, \"sendmean\": %f, \"sendmin\": %f, \"sendmax\": %f, \"sendstddev\": %f, \"recvcount\": %f, \"recvmean\": %f, \"recvmin\": %f, \"recvmax\": %f, \"recvsumsqr\": %f, \"comm\": \"%s\"", \
    EVENT_TRACE_PREFIX, __desc, __stats1[0],__stats1[1],__stats1[2],__stats1[3],__stats1[4], \
    __stats2[0],__stats2[1],__stats2[2],__stats2[3],__stats2[4], __commstr); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COMM_SPLIT_EVENT(__comm,__color,__key,__comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Comm_split\", \"comm_in\": \"%s\", \"color\": %d, \"key\": %d, \"comm_out\": \"0x%" PRIx64 "\"", EVENT_TRACE_PREFIX, __commstr,__color,__key,(uint64_t)__comm_out); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COMM_DUP_EVENT(__comm,__comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Comm_dup\", \"comm_in\": \"%s\", \"comm_out\": \"0x%" PRIx64 "\"", EVENT_TRACE_PREFIX, __commstr, (uint64_t)__comm_out); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COMM_FREE_EVENT(__comm) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Comm_free\", \"comm\": \"%s\"", EVENT_TRACE_PREFIX, __commstr); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COMM_CREATE_EVENT(__comm,__group,__comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Comm_create\", \"comm_in\": \"%s\", \"group\": \"%p\", \"comm_out\": \"0x%" PRIx64 "\"", EVENT_TRACE_PREFIX, __commstr, (void*)__group, (uint64_t)__comm_out); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_COMM_GROUP_EVENT(__comm,__group_addr) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __commstr[64]; \
convert_comm(__commstr, __comm); \
char __tmp[1024]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Comm_group\", \"comm\": \"%s\", \"group_addr\": \"%p\"", EVENT_TRACE_PREFIX, __commstr, (void*)__group_addr); \
Tau_plugin_trace_current_timer(__tmp); \
}

void Tau_timer_exit_group_incl_event(MPI_Group group, int count, const int ranks[], MPI_Group new_group) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(count*11), sizeof(char)));
    snprintf(tmp, "%s \"MPI_Group_incl\", \"group\": \"%p\", \"count\": %d, \"ranks\": [", EVENT_TRACE_PREFIX, (void*)group,count);
    int x;
    for (x = 0 ; x < count-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, ranks[x]);
    }
    sprintf(tmp, "%s%d], \"new_group\": \"%p\"", tmp, ranks[count-1], (void*)new_group);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_GROUP_INCL_EVENT(__group,__count,__ranks,__group_addr) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_group_incl_event(__group, __count, __ranks, __group_addr); \
}

void Tau_timer_exit_group_excl_event(MPI_Group group, int count, const int ranks[], MPI_Group new_group) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(count*11), sizeof(char)));
    sprintf(tmp, "%s \"MPI_Group_excl\", \"group\": \"%p\", \"count\": %d, \"ranks\": [", EVENT_TRACE_PREFIX, (void*)group,count);
    int x;
    for (x = 0 ; x < count-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, ranks[x]);
    }
    sprintf(tmp, "%s%d], \"new_group\": \"%p\"", tmp, ranks[count-1], (void*)new_group);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_GROUP_EXCL_EVENT(__group,__count,__ranks,__group_addr) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_group_excl_event(__group, __count, __ranks, __group_addr); \
}

void Tau_timer_exit_group_range_incl_event(MPI_Group group, int count, const int ranges[][3], MPI_Group new_group) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(count*33), sizeof(char)));
    sprintf(tmp, "%s \"MPI_Group_range_incl\", \"group\": \"%p\", \"count\": %d, \"ranges\": [", EVENT_TRACE_PREFIX, (void*)group,count);
    int x;
    for (x = 0 ; x < count-1 ; x++ ) {
        sprintf(tmp, "%s[%d,%d,%d],", tmp, ranges[x][0], ranges[x][1], ranges[x][2]);
    }
    sprintf(tmp, "%s[%d,%d,%d]], \"new_group\": \"%p\"", tmp, ranges[count-1][0], ranges[count-1][1], ranges[count-1][2], (void*)new_group);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_GROUP_RANGE_INCL_EVENT(__group,__count,__ranges,__newgroup) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_group_range_incl_event(__group, __count, __ranges, __newgroup); \
}

void Tau_timer_exit_group_range_excl_event(MPI_Group group, int count, int ranges[][3], MPI_Group new_group) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(count*33), sizeof(char)));
    sprintf(tmp, "%s \"MPI_Group_range_excl\", \"group\": \"%p\", \"count\": %d, \"ranges\": [", EVENT_TRACE_PREFIX, (void*)group,count);
    int x;
    for (x = 0 ; x < count-1 ; x++ ) {
        sprintf(tmp, "%s[%d,%d,%d],", tmp, ranges[x][0], ranges[x][1], ranges[x][2]);
    }
    sprintf(tmp, "%s[%d,%d,%d]], \"new_group\": \"%p\"", tmp, ranges[count-1][0], ranges[count-1][1], ranges[count-1][2], (void*)new_group);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_GROUP_RANGE_EXCL_EVENT(__group,__count,__ranges,__newgroup) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_group_range_excl_event(__group, __count, __ranges, __newgroup); \
}

void Tau_timer_exit_group_translate_ranks_event(MPI_Group group1, int count, const int *ranks1, MPI_Group group2, int *ranks2) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(count*11), sizeof(char)));
    sprintf(tmp, "%s \"MPI_Group_translate_ranks\", \"group_in\": \"%p\", \"count\": %d, \"ranks_in\": [", EVENT_TRACE_PREFIX, (void*)group1, count);
    int x;
    for (x = 0 ; x < count-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, ranks1[x]);
    }
    sprintf(tmp, "%s%d], \"ranks_out\": [", tmp, ranks1[count-1]);
    for (x = 0 ; x < count-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, ranks2[x]);
    }
    sprintf(tmp, "%s%d], \"group_out\": \"%p\"", tmp, ranks2[count-1], (void*)group2);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_GROUP_TRANSLATE_RANKS_EVENT(__group,__count,__ranks1,__group2,__ranks2) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_group_translate_ranks_event(__group, __count, __ranks1, __group2, __ranks2); \
}

#define TIMER_EXIT_GROUP_DIFFERENCE_EVENT(__group1,__group2,__newgroup) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __tmp[256]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Group_difference\", \"group1\": \"%p\", \"group2\": \"%p\", \"new_group\": \"%p\"", EVENT_TRACE_PREFIX, (void*)__group1, (void*)__group2, (void*)__newgroup); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_GROUP_INTERSECTION_EVENT(__group1,__group2,__newgroup) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __tmp[256]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Group_intersection\", \"group1\": \"%p\", \"group2\": \"%p\", \"new_group\": \"%p\"", EVENT_TRACE_PREFIX, (void*)__group1, (void*)__group2, (void*)__newgroup); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_GROUP_UNION_EVENT(__group1,__group2,__newgroup) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __tmp[256]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Group_union\", \"group1\": \"%p\", \"group2\": \"%p\", \"new_group\": \"%p\"", EVENT_TRACE_PREFIX, (void*)__group1, (void*)__group2, (void*)__newgroup); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_INTERCOMM_CREATE_EVENT(__local_comm, __local_leader, __peer_comm, __remote_leader, __tag, __comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __local_commstr[64]; \
char __peer_commstr[64]; \
convert_comm(__local_commstr, __local_comm); \
convert_comm(__peer_commstr, __peer_comm); \
char __tmp[256]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Intercomm_create\", \"local_comm\": \"%s\", \"local_leader\": \"%d\", \"peer_comm\": \"%s\", \"remote_leader\": \"%d\", \"tag\": \"%d\", \"comm_out\": \"0x%" PRIx64 "\"", EVENT_TRACE_PREFIX, __local_commstr, __local_leader, __peer_commstr, __remote_leader, __tag, (uint64_t)__comm_out); \
Tau_plugin_trace_current_timer(__tmp); \
}

#define TIMER_EXIT_INTERCOMM_MERGE_EVENT(__local_comm, __high, __comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
char __local_commstr[64]; \
convert_comm(__local_commstr, __local_comm); \
char __tmp[256]; \
snprintf(__tmp, sizeof(__tmp),  "%s \"MPI_Intercomm_merge\", \"local_comm\": \"%s\", \"high\": \"%d\", \"comm_out\": \"0x%" PRIx64 "\"", EVENT_TRACE_PREFIX, __local_commstr, __high, (uint64_t)__comm_out); \
Tau_plugin_trace_current_timer(__tmp); \
}

// this is used between cart_create and cart_sub calls... may not be safe, but...
static int __cart_dims = 1;

void Tau_timer_exit_cart_create_event(MPI_Comm comm, int ndims, TAU_MPICH3_CONST int * dims, TAU_MPICH3_CONST int * periods, int reorder, MPI_Comm comm_out) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(ndims*22), sizeof(char)));
    char commstr[64];
    convert_comm(commstr, comm);
    sprintf(tmp, "%s \"MPI_Cart_create\", \"comm\": \"%s\", \"ndims\": %d, \"dims\": [", EVENT_TRACE_PREFIX, commstr, ndims);
    int x;
    __cart_dims = ndims;
    for (x = 0 ; x < ndims-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, dims[x]);
    }
    sprintf(tmp, "%s%d], \"periods\": [", tmp, dims[ndims-1]);
    for (x = 0 ; x < ndims-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, periods[x]);
    }
    sprintf(tmp, "%s%d], \"reorder\": %d, \"comm_out\": \"0x%" PRIx64 "\"", tmp, periods[ndims-1], reorder, (uint64_t)comm_out);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_CART_CREATE_EVENT(__comm,__ndims,__dims,__periods,__reorder,__comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
   Tau_timer_exit_cart_create_event(__comm,__ndims,__dims,__periods,__reorder,__comm_out); \
}

void Tau_timer_exit_cart_coords_event(MPI_Comm comm, int rank, int maxdims, int * coords) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(maxdims*11), sizeof(char)));
    char commstr[64];
    convert_comm(commstr, comm);
    sprintf(tmp, "%s \"MPI_Cart_coords\", \"comm\": \"%s\", \"rank\": %d, \"maxdims\": %d, \"coords\": [", EVENT_TRACE_PREFIX, commstr,rank,maxdims);
    int x;
    for (x = 0 ; x < maxdims-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, coords[x]);
    }
    sprintf(tmp, "%s%d]", tmp, coords[maxdims-1]);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_CART_COORDS_EVENT(__comm,__rank,__maxdims,__coords) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_cart_coords_event(__comm,__rank,__maxdims,__coords); \
}

void Tau_timer_exit_cart_sub_event(MPI_Comm comm, TAU_MPICH3_CONST int * remains, MPI_Comm comm_out) {
    // assume 128 for letters, and 10 digits for each rank (plus a comma)
    char * tmp = (char*)(calloc(128+(__cart_dims*11), sizeof(char)));
    char commstr[64];
    convert_comm(commstr, comm);
    sprintf(tmp, "%s \"MPI_Cart_sub\", \"comm\": \"%s\", \"remains\": [", EVENT_TRACE_PREFIX, commstr);
    int x;
    for (x = 0 ; x < __cart_dims-1 ; x++ ) {
        sprintf(tmp, "%s%d,", tmp, remains[x]);
    }
    sprintf(tmp, "%s%d], \"comm_out\": \"0x%" PRIx64 "\"", tmp, remains[__cart_dims-1], (uint64_t)comm_out);
    Tau_plugin_trace_current_timer(tmp);
    free(tmp);
}

#define TIMER_EXIT_CART_SUB_EVENT(__comm,__remains,__comm_out) \
if(Tau_plugins_enabled.current_timer_exit && TAU_DO_TIMER_EXIT) { \
    Tau_timer_exit_cart_sub_event(__comm,__remains,__comm_out); \
}

#else

#define TIMER_EXIT_COLLECTIVE_SYNC_EVENT(__desc,__comm)
#define TIMER_EXIT_COLLECTIVE_EXCH_EVENT(__desc,__size,__root,__comm)
#define TIMER_EXIT_COLLECTIVE_EXCH_ALL_EVENT(__desc,__send_size,__recv_size,__root,__comm)
#define TIMER_EXIT_COLLECTIVE_EXCH_V_EVENT(__desc,__label,__mybytes,__stats,__root,__comm)
#define TIMER_EXIT_COLLECTIVE_EXCH_AAV_EVENT(__desc,__stats1,__stats2,__comm)
#define TIMER_EXIT_COMM_SPLIT_EVENT(__comm,__color,__key,__comm_out)
#define TIMER_EXIT_COMM_DUP_EVENT(__comm,__comm_out)
#define TIMER_EXIT_COMM_FREE_EVENT(__comm)
#define TIMER_EXIT_COMM_CREATE_EVENT(__comm,__group,__comm_out)
#define TIMER_EXIT_COMM_GROUP_EVENT(__comm,__group_addr)
#define TIMER_EXIT_GROUP_INCL_EVENT(__group,__count,__ranks,__group_addr)
#define TIMER_EXIT_GROUP_EXCL_EVENT(__group,__count,__ranks,__group_addr)
#define TIMER_EXIT_GROUP_RANGE_INCL_EVENT(__group,__count,__ranges,__newgroup)
#define TIMER_EXIT_GROUP_RANGE_EXCL_EVENT(__group,__count,__ranges,__newgroup)
#define TIMER_EXIT_GROUP_TRANSLATE_RANKS_EVENT(__group,__count,__ranks1,__group2,__ranks2)
#define TIMER_EXIT_GROUP_DIFFERENCE_EVENT(__group1,__group2,__newgroup)
#define TIMER_EXIT_GROUP_INTERSECTION_EVENT(__group1,__group2,__newgroup)
#define TIMER_EXIT_GROUP_UNION_EVENT(__group1,__group2,__newgroup)
#define TIMER_EXIT_INTERCOMM_CREATE_EVENT(__local_comm, __local_leader, __peer_comm, __remote_leader, __tag, __comm_out)
#define TIMER_EXIT_INTERCOMM_MERGE_EVENT(__local_comm, __high, __comm_out)
#define TIMER_EXIT_CART_CREATE_EVENT(__comm,__ndims,__dims,__periods,__reorder,__comm_out)
#define TIMER_EXIT_CART_COORDS_EVENT(__comm,__rank,__maxdims,__coords)
#define TIMER_EXIT_CART_SUB_EVENT(__comm,__remains,__comm_out)


#endif /* defined(TAU_SOS) || defined(TAU_ADIOS2) */
