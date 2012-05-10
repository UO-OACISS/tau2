#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef TAUDB_PERFDMF
PERFDMF_APPLICATION* perfdmf_create_applications(int count){ 
 PERFDMF_APPLICATION* applications = (PERFDMF_APPLICATION*) (calloc (count, sizeof (PERFDMF_APPLICATION)));
 return applications;
}

PERFDMF_EXPERIMENT* perfdmf_create_experiments(int count){ 
 PERFDMF_EXPERIMENT* experiments = (PERFDMF_EXPERIMENT*) (calloc (count, sizeof (PERFDMF_EXPERIMENT)));
 return experiments;
}
#endif

TAUDB_CONFIGURATION* taudb_create_configuration(){ 
 TAUDB_CONFIGURATION* config = (TAUDB_CONFIGURATION*) (calloc (1, sizeof (TAUDB_CONFIGURATION)));
 return config;
}

TAUDB_TRIAL* taudb_create_trials(int count){ 
 TAUDB_TRIAL* trials = (TAUDB_TRIAL*) (calloc (count, sizeof (TAUDB_TRIAL)));
 return trials;
}

TAUDB_METRIC* taudb_create_metrics(int count){ 
 TAUDB_METRIC* metrics = (TAUDB_METRIC*) (calloc (count, sizeof (TAUDB_METRIC)));
 return metrics;
}

TAUDB_THREAD* taudb_create_threads(int count){ 
 TAUDB_THREAD* threads = (TAUDB_THREAD*) (calloc (count, sizeof (TAUDB_THREAD)));
 return threads;
}

TAUDB_SECONDARY_METADATA* taudb_create_secondary_metadata(int count){ 
 TAUDB_SECONDARY_METADATA* metadata = (TAUDB_SECONDARY_METADATA*) (calloc (count, sizeof (TAUDB_SECONDARY_METADATA)));
 return metadata;
}

TAUDB_PRIMARY_METADATA* taudb_create_primary_metadata(int count){ 
 TAUDB_PRIMARY_METADATA* metadata = (TAUDB_PRIMARY_METADATA*) (calloc (count, sizeof (TAUDB_PRIMARY_METADATA)));
 return metadata;
}

TAUDB_PRIMARY_METADATA* taudb_resize_primary_metadata(int count, TAUDB_PRIMARY_METADATA* old_primary_metadata){ 
 TAUDB_PRIMARY_METADATA* primary_metadata = (TAUDB_PRIMARY_METADATA*) (realloc (old_primary_metadata, count * sizeof (TAUDB_PRIMARY_METADATA)));
 return primary_metadata;
}

TAUDB_COUNTER* taudb_create_counters(int count){ 
 TAUDB_COUNTER* counters = (TAUDB_COUNTER*) (calloc (count, sizeof (TAUDB_COUNTER)));
 int i = 0;
 for (i = 0 ; i < count ; i++) {
   counters[i].group_count = 0;
   counters[i].value_count = 0;
 }
 return counters;
}

TAUDB_COUNTER_GROUP* taudb_create_counter_groups(int count){ 
 TAUDB_COUNTER_GROUP* groups = (TAUDB_COUNTER_GROUP*) (calloc (count, sizeof (TAUDB_COUNTER_GROUP)));
 return groups;
}

TAUDB_COUNTER_GROUP* taudb_resize_counter_groups(int count, TAUDB_COUNTER_GROUP* old_groups){ 
 TAUDB_COUNTER_GROUP* counter_groups = (TAUDB_COUNTER_GROUP*) (realloc (old_groups, count * sizeof (TAUDB_COUNTER_GROUP)));
 return counter_groups;
}

TAUDB_COUNTER_VALUE* taudb_create_counter_values(int count){ 
 TAUDB_COUNTER_VALUE* counter_values = (TAUDB_COUNTER_VALUE*) (calloc (count, sizeof (TAUDB_COUNTER_VALUE)));
 return counter_values;
}

TAUDB_TIMER* taudb_create_timers(int count){ 
 TAUDB_TIMER* timers = (TAUDB_TIMER*) (calloc (count, sizeof (TAUDB_TIMER)));
 int i = 0;
 for (i = 0 ; i < count ; i++) {
   timers[i].child_count = 0;
   timers[i].group_count = 0;
   timers[i].parameter_count = 0;
 }
 return timers;
}

TAUDB_TIMER_PARAMETER* taudb_create_timer_parameters(int count){ 
 TAUDB_TIMER_PARAMETER* timer_parameters = (TAUDB_TIMER_PARAMETER*) (calloc (count, sizeof (TAUDB_TIMER_PARAMETER)));
 return timer_parameters;
}

TAUDB_TIMER_GROUP* taudb_create_timer_groups(int count){ 
 TAUDB_TIMER_GROUP* timer_groups = (TAUDB_TIMER_GROUP*) (calloc (count, sizeof (TAUDB_TIMER_GROUP)));
 return timer_groups;
}

TAUDB_TIMER_GROUP* taudb_resize_timer_groups(int count, TAUDB_TIMER_GROUP* old_groups){ 
 TAUDB_TIMER_GROUP* timer_groups = (TAUDB_TIMER_GROUP*) (realloc (old_groups, count * sizeof (TAUDB_TIMER_GROUP)));
 return timer_groups;
}

TAUDB_TIMER_VALUE* taudb_create_timer_values(int count){ 
 TAUDB_TIMER_VALUE* timer_values = (TAUDB_TIMER_VALUE*) (calloc (count, sizeof (TAUDB_TIMER_VALUE)));
 return timer_values;
}

TAUDB_TIMER_CALLPATH* taudb_create_timer_callpaths(int count){ 
 TAUDB_TIMER_CALLPATH* timer_callpaths = (TAUDB_TIMER_CALLPATH*) (calloc (count, sizeof (TAUDB_TIMER_CALLPATH)));
 return timer_callpaths;
}


