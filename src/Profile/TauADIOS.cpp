/* 
 * ADIOS is freely available under the terms of the BSD license described
 * in the COPYING file in the top level directory of this source distribution.
 *
 * Copyright (c) 2008 - 2009.  UT-BATTELLE, LLC. All rights reserved.
 */

/* ADIOS Event Callback API - TAU Tool implementation
 *
 * This source file is a TAU implementation of the ADIOS callback API
 * for tools, or ADIOST.
 */

#ifdef TAU_SOS
#include "Profile/TauPluginInternals.h"
#endif
#include "Profile/Profiler.h"
#include "Profile/UserEvent.h"
#include "Profile/TauMetrics.h"

#include "adiost_callback_api.h"
#include "adios_types.h"
#include "mpi.h"
#include <stdint.h>
#include <sstream>
#include <iostream>
#include <algorithm>
#define ADIOST_EXTERN extern "C"

/* We need a thread-local static stack of ADIOS API calls 
   in order to handle the trace events correctly */
#if defined (TAU_USE_TLS)
__thread int function_stack;
#elif defined (TAU_USE_DTLS)
__declspec(thread) int function_stack;
#elif defined (TAU_USE_PGS)
#include "pthread.h"
pthread_key_t thr_id_key;
#endif

/* These macros are so we can compile out the SOS support */

#ifdef TAU_SOS

#define EVENT_TRACE_PREFIX "TAU_EVENT::"

int Tau_increment_stack_height() {
    // get the current API call stack
#if defined (TAU_USE_TLS) || defined (TAU_USE_DTLS)
	function_stack = function_stack+1;
	return function_stack;
#else
	int function_stack = pthread_getspecific(thr_id_key);
	pthread_setspecific(thr_id_key, function_stack + 1);
	return function_stack;
#endif
}

int TAU_decrement_stack_height() {
    // get the current API call stack
#if defined (TAU_USE_TLS) || defined (TAU_USE_DTLS)
	function_stack = function_stack-1;
	return function_stack;
#else
	int function_stack = pthread_getspecific(thr_id_key);
	pthread_setspecific(thr_id_key, function_stack - 1);
	return function_stack;
#endif
}

/* Because we are collecting an API trace, we don't want to trace the 
   internal ADIOS calls.  So keep track of the ADIOS stack depth, and only
   output a trace event if we aren't currently timing another ADIOS call. */

void Tau_SOS_conditionally_pack_current_timer(const char * name) {
    int foo = TAU_decrement_stack_height();
    if (foo == 0) {
        /*Invoke plugins only if both plugin path and plugins are specified*/
        if(TauEnv_get_plugins_enabled()) {
            Tau_plugin_event_current_timer_exit_data plugin_data;
            plugin_data.name_prefix = name;
            Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT, &plugin_data);
        }
	}
}

#define TAU_SOS_COLLECTIVE_ADIOS_EVENT(__detail) \
        std::stringstream __ss; \
        __ss << EVENT_TRACE_PREFIX << __detail << "()"; \
        Tau_SOS_conditionally_pack_current_timer(__ss.str().c_str());

void TAU_SOS_collective_ADIOS_write_event(const char * detail, 
    const char * var_name, enum ADIOS_DATATYPES data_type, 
    const int ndims, const char * dims, const void * value) {
    std::stringstream ss;
    ss << EVENT_TRACE_PREFIX << detail << "(" << var_name << ",";
    switch(data_type) {
        case adios_byte:
            ss << "adios_byte" ; break;
        case adios_short:
            ss << "adios_short" ; break;
        case adios_integer:
            ss << "adios_integer" ; break;
        case adios_long:
            ss << "adios_long" ; break;
        case adios_unsigned_byte:
            ss << "adios_unsigned_byte" ; break;
        case adios_unsigned_short:
            ss << "adios_unsigned_short" ; break;
        case adios_unsigned_integer:
            ss << "adios_unsigned_integer" ; break;
        case adios_unsigned_long:
            ss << "adios_unsigned_long" ; break;
        case adios_real:
            ss << "adios_real" ; break;
        case adios_double:
            ss << "adios_double" ; break;
        case adios_long_double:
            ss << "adios_long_double" ; break;
        case adios_complex:
            ss << "adios_complex" ; break;
        case adios_double_complex:
            ss << "adios_double_complex" ; break;
        case adios_string:
            ss << "adios_string" ; break;
    }
    ss << "," << ndims << ",";
    if (ndims == 0) {
        ss << "[" << dims << "],";
        switch(data_type) {
            case adios_byte:
                ss << *(char*)(value) ; break;
            case adios_short:
                ss << *(short*)(value) ; break;
            case adios_integer:
                ss << *(int*)(value) ; break;
            case adios_long:
                ss << *(long*)(value) ; break;
            case adios_unsigned_byte:
                ss << *(unsigned char*)(value) ; break;
            case adios_unsigned_short:
                ss << *(unsigned short*)(value) ; break;
            case adios_unsigned_integer:
                ss << *(unsigned int*)(value) ; break;
            case adios_unsigned_long:
                ss << *(unsigned long*)(value) ; break;
            //case adios_real:
                //ss << *(float*)(value) ; break;
            //case adios_double:
                //ss << *(double*)(value) ; break;
            //case adios_long_double:
                //ss << *(long double*)(value) ; break;
            //case adios_complex:
            //case adios_double_complex:
            //case adios_string:
            default:
                ss << "0";
                break;
        }
    } else {
        ss << "[" << dims << "]";
    }
    ss << ")";
	//printf("%s\n", ss.str().c_str());
    Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
}
#else
#define TAU_SOS_COLLECTIVE_ADIOS_EVENT // do nuthin.
#define Tau_increment_stack_height()
#define TAU_decrement_stack_height()
#define EVENT_TRACE_PREFIX ""
#define Tau_SOS_conditionally_pack_current_timer(...)
#define TAU_SOS_collective_ADIOS_write_event(...)
#endif

ADIOST_EXTERN void tau_adiost_thread ( adiost_event_type_t type, 
		int64_t file_descriptor,
    const char * thread_name) {
    if (type == adiost_event_enter) {
        Tau_register_thread();
        Tau_create_top_level_timer_if_necessary();
        Tau_pure_start_task(thread_name, Tau_get_thread());
	} else if (type == adiost_event_exit) {
        Tau_pure_stop_task(thread_name, Tau_get_thread());
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_init ( adiost_event_type_t type, 
	const char * xml_fname, MPI_Comm comm) {
	const char * function_name = "adios_init";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	} else if (type == adiost_event_exit) {
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_finalize(adiost_event_type_t type, int proc_id) {
	const char * function_name = "adios_finalize";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_open ( adiost_event_type_t type, 
		int64_t file_descriptor,
    const char * group_name, const char * file_name, const char * mode,
	MPI_Comm comm) {
	const char * function_name = "adios_open";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_close(adiost_event_type_t type, 
		int64_t file_descriptor) {
	const char * function_name = "adios_close";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_write( adiost_event_type_t type, 
		int64_t file_descriptor, const char * name, 
		enum ADIOS_DATATYPES data_type, const int ndims, 
		const char * dims, const void * value) {
	const char * function_name = "adios_write";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
#if defined(TAU_SOS)
        TAU_SOS_collective_ADIOS_write_event(function_name, name, data_type, ndims, dims, value);
#endif
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_write_byid( adiost_event_type_t type, 
		int64_t file_descriptor, const char * name, 
		enum ADIOS_DATATYPES data_type, const int ndims, 
		const char * dims, const void * value) {
	const char * function_name = "adios_write_byid";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
#if defined(TAU_SOS)
        TAU_SOS_collective_ADIOS_write_event(function_name, name, data_type, ndims, dims, value);
#endif
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_read( adiost_event_type_t type, 
		int64_t file_descriptor, const char * name, const void * buffer, uint64_t buffer_size) {
	const char * function_name = "adios_read";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_group_size(adiost_event_type_t type, 
		int64_t file_descriptor, 
    	uint64_t data_size, uint64_t total_size) {
    TAU_REGISTER_CONTEXT_EVENT(c1, "ADIOS data size");
    TAU_REGISTER_CONTEXT_EVENT(c2, "ADIOS total size");
	const char * function_name = "adios_group_size";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_CONTEXT_EVENT(c1, (double)data_size);
        TAU_CONTEXT_EVENT(c2, (double)total_size);
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_transform( adiost_event_type_t type, 
		int64_t file_descriptor) {
	const char * function_name = "adios_group_size";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_fp_send_read_msg(adiost_event_type_t type, 
		int64_t file_descriptor) { 
	const char * function_name = "adios_fp_send_read_msg";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_fp_send_finalize_msg(adiost_event_type_t type, 
		int64_t file_descriptor) { 
	const char * function_name = "adios_fp_send_finalize_msg";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_fp_add_var_to_read_msg(adiost_event_type_t type, 
		int64_t file_descriptor) { 
	const char * function_name = "adios_fp_add_var_to_read_msg";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_fp_copy_buffer(adiost_event_type_t type, 
		int64_t file_descriptor) { 
	const char * function_name = "adios_fp_copy_buffer";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

/*
 * ------------------ Special events for No-XML Write API -------------------- *
 */

ADIOST_EXTERN void tau_adiost_init_noxml(adiost_event_type_t type, 
		MPI_Comm comm) { 
	const char * function_name = "adios_init_noxml";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_set_max_buffer_size(adiost_event_type_t type, 
		uint64_t buffer_size) { 
	const char * function_name = "adios_set_max_buffer_size";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_declare_group(adiost_event_type_t type, 
		const int64_t * id, const char * name, const char * time_index,
		enum ADIOS_STATISTICS_FLAG stats) { 
	const char * function_name = "adios_declare_group";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var(adiost_event_type_t type, 
		int64_t group_id, const char * name, const char * path,
		enum ADIOS_DATATYPES data_type, const char * dimensions,
		const char * global_dimensions, const char * local_offsets) {
	const char * function_name = "adios_define_var";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_set_transform(adiost_event_type_t type, 
    int64_t var_id, const char * transform_type_str) {
	const char * function_name = "adios_set_transform";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_attribute(adiost_event_type_t type, 
    int64_t group, const char * name, const char * path,
    enum ADIOS_DATATYPES data_type, const char * value, const char * var) {
	const char * function_name = "adios_define_attribute";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_attribute_byvalue(adiost_event_type_t type, 
    int64_t group, const char * name, const char * path,
    enum ADIOS_DATATYPES data_type, int nelems, const char * value) {
	const char * function_name = "adios_define_attribute_byvalue";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_select_method(adiost_event_type_t type, 
    int64_t group, const char * method,
    const char * parameters, const char * base_path){
	const char * function_name = "adios_select_method";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_expected_var_size(adiost_event_type_t type, 
    int64_t var_id) {
	const char * function_name = "adios_expected_var_size";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

/*
 * ------------- Special events for no-XML write for viz --------------- *
 */

ADIOST_EXTERN void tau_adiost_define_schema_version(adiost_event_type_t type, 
    int64_t group_id, const char * schema_version) {
	const char * function_name = "adios_define_schema_version";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var_mesh(adiost_event_type_t type, 
    int64_t group_id, const char * varname, const char * meshname) {
	const char * function_name = "adios_define_var_mesh";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var_centering(adiost_event_type_t type, 
    int64_t group_id, const char * varname, const char * centering) {
	const char * function_name = "adios_define_var_centering";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var_timesteps(adiost_event_type_t type, 
    const char * timesteps, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_var_timesteps";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var_timescale(adiost_event_type_t type, 
    const char * timescale, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_var_timescale";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var_timeseriesformat(adiost_event_type_t type, 
    const char * timeseries, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_var_timeseriesformat";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_var_hyperslab(adiost_event_type_t type, 
    const char * hyperslab, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_var_hyperslab";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_timevarying(adiost_event_type_t type, 
    const char * timevarying, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_timevarying";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_timesteps(adiost_event_type_t type, 
    const char * timesteps, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_timesteps";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_timescale(adiost_event_type_t type, 
    const char * timescale, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_timescale";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_timeseriesformat(adiost_event_type_t type, 
    const char * timeseries, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_timeseriesformat";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_group(adiost_event_type_t type, 
    const char * group, int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_group";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_file(adiost_event_type_t type, 
    int64_t group_id, const char * name, const char * file) {
	const char * function_name = "adios_define_mesh_file";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_uniform(adiost_event_type_t type, 
    const char * dimensions, const char * origin, const char * spacing,
	const char * maximum, const char * nspace,
    int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_uniform";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_rectilinear(adiost_event_type_t type, 
    const char * dimensions, const char * coordinates, const char * nspace,
    int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_rectilinear";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_structured(adiost_event_type_t type, 
    const char * dimensions, const char * points, const char * nspace,
    int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_structured";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

ADIOST_EXTERN void tau_adiost_define_mesh_unstructured(adiost_event_type_t type, 
    const char * points, const char * data, const char * count,
    const char * cell_type, const char * npoints, const char * nspace,
    int64_t group_id, const char * name) {
	const char * function_name = "adios_define_mesh_unstructured";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
	} else if (type == adiost_event_exit) {
        TAU_SOS_COLLECTIVE_ADIOS_EVENT(function_name);
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
} 

/*
 * ------------------ Special events for Read API -------------------- *
 */

ADIOST_EXTERN void tau_adiost_read_init_method(
    adiost_event_type_t type,
    enum ADIOS_READ_METHOD method, 
    MPI_Comm comm, 
    const char * parameters) {
	const char * function_name = "adios_read_init_method";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " method: " << method << ",";
	ss << " comm: " << std::hex << "0x" << comm << ",";
	// The parameters has newlines in it - strip them out.
	std::string s(parameters);
	std::replace(s.begin(), s.end(), '\n', ' ');
	std::replace(s.begin(), s.end(), '\r', ' ');
	// The parameters are corrupting the SQL insert later, disabled for now
	//ss << " parameters: [" << s.c_str() << "]";
	ss << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
	    TAU_PROFILE_STOP(tautimer);
	    Tau_increment_stack_height();
    } else {
	    // not conditional! neither start nor stop.
        /*Invoke plugins only if both plugin path and plugins are specified*/
        if(TauEnv_get_plugins_enabled()) {
            Tau_plugin_event_current_timer_exit_data plugin_data;
            plugin_data.name_prefix = ss.str().c_str();
            Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT, &plugin_data);
        }
    }
}

ADIOST_EXTERN void tau_adiost_read_finalize_method(
    adiost_event_type_t type,
    enum ADIOS_READ_METHOD method ) {
	const char * function_name = "adios_read_finalize_method";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " method: " << method << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
	    TAU_PROFILE_STOP(tautimer);
	    Tau_increment_stack_height();
    } else {
        /*Invoke plugins only if both plugin path and plugins are specified*/
        if(TauEnv_get_plugins_enabled()) {
            Tau_plugin_event_current_timer_exit_data plugin_data;
            plugin_data.name_prefix = ss.str().c_str();
            Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT, &plugin_data);
        }
    }
}

ADIOST_EXTERN void tau_adiost_read_open(
    adiost_event_type_t type,
    enum ADIOS_READ_METHOD method, 
    MPI_Comm comm, 
	enum ADIOS_LOCKMODE lock_mode,
    float timeout_sec,
	ADIOS_FILE * file_descriptor) {
	const char * function_name = "adios_read_open";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " method: " << method << ",";
	ss << " comm: " << std::hex << "0x" << comm << ",";
	ss << " lock_mode: " << lock_mode << ",";
	ss << " timeout_sec: " << timeout_sec << ",";
	ss << " file_descriptor: " << std::hex << file_descriptor << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_read_open_file(
    adiost_event_type_t type,
    const char * fname,
    enum ADIOS_READ_METHOD method, 
    MPI_Comm comm,
	ADIOS_FILE * file_descriptor) {
	const char * function_name = "adios_read_open_file";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
   	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fname: '" << fname << "',";
	ss << " method: " << method << ",";
	ss << " comm: " << std::hex << "0x" << comm << ",";
	ss << " file_descriptor: " << std::hex << file_descriptor << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_advance_step(
    adiost_event_type_t type,
    ADIOS_FILE *fp,
    int last,
    float timeout_sec) {
	const char * function_name = "adios_advance_step";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fp: " << std::hex << fp << ",";
	ss << " last: " << last << ",";
	ss << " timeout_sec: " << timeout_sec << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
	    TAU_PROFILE_STOP(tautimer);
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_inq_var(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
    const char * varname,
	ADIOS_VARINFO * varinfo) {
	const char * function_name = "adios_inq_var";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fp: " << std::hex << fp << ",";
	ss << " varname: '" << varname << "',";
	ss << " varinfo: " << std::hex << varinfo << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_inq_var_byid(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
    int varid,
	ADIOS_VARINFO * varinfo) {
	const char * function_name = "adios_inq_var_byid";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fp: " << std::hex << fp << ",";
	ss << " varid: " << varid << ",";
	ss << " varinfo: " << std::hex << varinfo << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_free_varinfo(
    adiost_event_type_t type,
	ADIOS_VARINFO * varinfo) {
	const char * function_name = "adios_free_varinfo";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " varinfo: " << std::hex << varinfo << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_inq_var_stat(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	ADIOS_VARINFO * varinfo,
	int per_prep_stat,
	int per_block_stat) {
	const char * function_name = "adios_inq_var_stat";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_inq_var_blockinfo(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	ADIOS_VARINFO * varinfo) {
	const char * function_name = "adios_inq_var_blockinfo";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_selection_boundingbox(
    adiost_event_type_t type,
    uint64_t ndim,
    const uint64_t *start,
    const uint64_t *count,
	ADIOS_SELECTION * selection) {
	const char * function_name = "adios_selection_boundingbox";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
   	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " ndim: " << ndim << ", start: [";
	int i;
	if (ndim > 0) {
		ss << start[0];
	}
	for (i = 1 ; i < ndim ; i++) {
		ss << "," << start[i];
    }
	ss << "], end: [";
	if (ndim > 0) {
		ss << count[0];
	}
	for (i = 1 ; i < ndim ; i++) {
		ss << "," << count[i];
    }
	ss << "],";
	ss << " selection: " << std::hex << selection << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_selection_points(
    adiost_event_type_t type,
    uint64_t ndim,
    uint64_t npoints,
    const uint64_t *points,
	ADIOS_SELECTION * container,
	int free_points_on_delete,
	ADIOS_SELECTION * selection) {
	const char * function_name = "adios_selection_points";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_selection_writeblock(
    adiost_event_type_t type,
    int index,
	ADIOS_SELECTION * selection) {
	const char * function_name = "adios_selection_writeblock";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_selection_auto(
    adiost_event_type_t type,
    char * hints,
	ADIOS_SELECTION * selection) {
	const char * function_name = "adios_selection_auto";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_selection_delete(
    adiost_event_type_t type,
	ADIOS_SELECTION * selection) {
	const char * function_name = "adios_selection_delete";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
   	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " selection: " << std::hex << selection << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_schedule_read(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	const ADIOS_SELECTION * selection,
	const char * varname,
	int from_steps,
	int nsteps,
	const char * param,
	void * data) {
	const char * function_name = "adios_schedule_read";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
   	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fp: " << std::hex << fp << ",";
	ss << " selection: " << std::hex << selection << ",";
	ss << " varname: '" << varname << "',";
	ss << " from_steps: " << from_steps << ",";
	ss << " nsteps: " << nsteps << ", param: '";
	if (param != NULL) {
		std::string s(param);
		std::replace(s.begin(), s.end(), '\n', ' ');
		std::replace(s.begin(), s.end(), '\r', ' ');
		std::replace(s.begin(), s.end(), ';', ',');
		ss << s.c_str();
	}
	ss << "',";
	ss << " data: " << std::hex << data << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_schedule_read_byid(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	const ADIOS_SELECTION * selection,
	int varid,
	int from_steps,
	int nsteps,
	const char * param,
	void * data) {
	const char * function_name = "adios_schedule_read_byid";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
   	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fp: " << std::hex << fp << ",";
	ss << " selection: " << std::hex << selection << ",";
	ss << " varid: " << varid << ",";
	ss << " from_steps: " << from_steps << ",";
	ss << " nsteps: " << nsteps << ", param: '";
	if (param != NULL) {
		std::string s(param);
		std::replace(s.begin(), s.end(), '\n', ' ');
		std::replace(s.begin(), s.end(), '\r', ' ');
		std::replace(s.begin(), s.end(), ';', ',');
		ss << s.c_str();
	}
	ss << "',";
	ss << " data: " << std::hex << data << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_perform_reads(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	int blocking) {
	const char * function_name = "adios_perform_reads";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
	std::stringstream ss;
	ss << EVENT_TRACE_PREFIX << function_name << "(";
	ss << " fp: " << std::hex << fp << ",";
	ss << " blocking: " << blocking << ")";
    if (type == adiost_event_enter) {
	    TAU_PROFILE_START(tautimer);
	    Tau_increment_stack_height();
    } else if (type == adiost_event_exit) {
		Tau_SOS_conditionally_pack_current_timer(ss.str().c_str());
	    TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_check_reads(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	ADIOS_VARCHUNK **chunk) {
	const char * function_name = "adios_check_reads";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_free_chunk(
    adiost_event_type_t type,
	ADIOS_VARCHUNK *chunk) {
	const char * function_name = "adios_free_chunk";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_get_attr(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	const char * attrname,
	enum ADIOS_DATATYPES * datatypes,
    int * size,
	void **data) {
	const char * function_name = "adios_get_attr";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_get_attr_byid(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	int attrid,
	enum ADIOS_DATATYPES * datatypes,
    int * size,
	void **data) {
	const char * function_name = "adios_get_attr_byid";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_type_to_string(
    adiost_event_type_t type,
    const char * name) {
	const char * function_name = "adios_type_to_string";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_type_size(
    adiost_event_type_t type,
    void * dadta,
	int size) {
	const char * function_name = "adios_type_size";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_get_grouplist(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	char ***group_namelist) {
	const char * function_name = "adios_get_grouplist";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_group_view(
    adiost_event_type_t type,
    ADIOS_FILE *fp,
	int groupid) {
	const char * function_name = "adios_group_view";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_stat_cov(
    adiost_event_type_t type,
    ADIOS_VARINFO * vix,
	ADIOS_VARINFO * viy,
	char * characteristic,
	uint32_t time_start,
	uint32_t time_end,
	uint32_t lag,
	double correlation) {
	const char * function_name = "adios_stat_cov";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_inq_mesh_byid(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	int meshid,
	ADIOS_MESH * mesh) {
	const char * function_name = "adios_inq_mesh_byid";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_free_meshinfo(
    adiost_event_type_t type,
	ADIOS_MESH * mesh) {
	const char * function_name = "adios_free_meshinfo";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

ADIOST_EXTERN void tau_adiost_inq_var_meshinfo(
    adiost_event_type_t type,
    const ADIOS_FILE *fp,
	ADIOS_VARINFO * varinfo) {
	const char * function_name = "adios_inq_var_meshinfo";
    TAU_PROFILE_TIMER(tautimer, function_name,  " ", TAU_IO);
    if (type == adiost_event_enter) {
        TAU_PROFILE_START(tautimer);
    } else if (type == adiost_event_exit) {
        TAU_PROFILE_STOP(tautimer);
    } else {
    }
}

/*
 * ------------------ Special events for Read API -------------------- *
 */

ADIOST_EXTERN void tau_adiost_library_shutdown(void) {
}

// This function is for checking that the function registration worked.
#define CHECK(EVENT,FUNCTION,NAME) \
    /*printf("TAU: Registering ADIOST callback %s...",NAME);*/ \
    fflush(stderr); \
    if (adiost_fn_set_callback(EVENT, (adiost_callback_t)(FUNCTION)) != \
                    adiost_set_result_registration_success) { \
        printf("\n\tFailed to register ADIOST callback %s!\n",NAME); \
        fflush(stderr); \
    } else { \
        /*printf("success.\n");*/ \
    } \

ADIOST_EXTERN void TAU_adiost_initialize (adiost_function_lookup_t adiost_fn_lookup,
    const char *runtime_version, unsigned int adiost_version) {

#if defined (TAU_USE_PGS)
    pthread_key_create(&thr_id_key, NULL);
    pthread_setspecific(thr_id_key, 0);
#endif

    adiost_set_callback_t adiost_fn_set_callback = 
        (adiost_set_callback_t)adiost_fn_lookup("adiost_set_callback");

    TAU_VERBOSE("Registering ADIOS tool events...\n");
	/* Special events */
    CHECK(adiost_event_thread,       tau_adiost_thread,        "adios_thread");
    CHECK(adiost_event_library_shutdown, tau_adiost_library_shutdown, "adios_library_shutdown");
	/* ADIOS Write API */
    CHECK(adiost_event_init,         tau_adiost_init,          "adios_init");
    CHECK(adiost_event_open,         tau_adiost_open,          "adios_open");
    CHECK(adiost_event_group_size,   tau_adiost_group_size,    "adios_group_size");
    CHECK(adiost_event_write,        tau_adiost_write,         "adios_write");
    CHECK(adiost_event_read,         tau_adiost_read,          "adios_read");
    CHECK(adiost_event_close,        tau_adiost_close,         "adios_close");
    CHECK(adiost_event_finalize,        tau_adiost_finalize,         "adios_finalize");
	/* ADIOS Flexpath (internal) events */
    CHECK(adiost_event_advance_step, tau_adiost_advance_step,  "adios_advance_step");
    CHECK(adiost_event_transform,    tau_adiost_transform,     "adios_transform");
    CHECK(adiost_event_fp_send_read_msg, 
        tau_adiost_fp_send_read_msg, "adios_fp_send_read_msg");
    CHECK(adiost_event_fp_send_finalize_msg, 
        tau_adiost_fp_send_finalize_msg, "adios_fp_send_finalize_msg");
    CHECK(adiost_event_fp_add_var_to_read_msg, 
        tau_adiost_fp_add_var_to_read_msg, "adios_fp_add_var_to_read_msg");
    CHECK(adiost_event_fp_copy_buffer, 
        tau_adiost_fp_copy_buffer, "adios_fp_copy_buffer");
	/* ADIOS No-XML Write API events */
    CHECK(adiost_event_init_noxml, tau_adiost_init_noxml, "adios_init_noxml");
    CHECK(adiost_event_set_max_buffer_size, tau_adiost_set_max_buffer_size, "adios_set_max_buffer_size");
    CHECK(adiost_event_declare_group, tau_adiost_declare_group, "adios_declare_group");
    CHECK(adiost_event_define_var, tau_adiost_define_var, "adios_define_var");
    CHECK(adiost_event_define_attribute, tau_adiost_define_attribute, "adios_define_attribute");
    CHECK(adiost_event_define_attribute_byvalue, tau_adiost_define_attribute_byvalue, "adios_define_attribute_byvalue");
    CHECK(adiost_event_write_byid, tau_adiost_write_byid, "adios_write_byid");
    CHECK(adiost_event_select_method, tau_adiost_select_method, "adios_select_method");
    CHECK(adiost_event_expected_var_size, tau_adiost_expected_var_size, "adios_expected_var_size");
	/* No-XML Write API for visualization schema Description */
    CHECK(adiost_event_define_schema_version, tau_adiost_define_schema_version, "adios_define_schema_version");
    CHECK(adiost_event_define_var_mesh, tau_adiost_define_var_mesh, "adios_define_var_mesh");
    CHECK(adiost_event_define_var_centering, tau_adiost_define_var_centering, "adios_define_var_centering");
    CHECK(adiost_event_define_var_timesteps, tau_adiost_define_var_timesteps, "adios_define_var_timesteps");
    CHECK(adiost_event_define_var_timescale, tau_adiost_define_var_timescale, "adios_define_var_timescale");
    CHECK(adiost_event_define_var_timeseriesformat, tau_adiost_define_var_timeseriesformat, "adios_define_var_timeseriesformat");
    CHECK(adiost_event_define_var_hyperslab, tau_adiost_define_var_hyperslab, "adios_define_var_hyperslab");
    CHECK(adiost_event_define_mesh_timevarying, tau_adiost_define_mesh_timevarying, "adios_define_mesh_timevarying");
    CHECK(adiost_event_define_mesh_timesteps, tau_adiost_define_mesh_timesteps, "adios_define_mesh_timesteps");
    CHECK(adiost_event_define_mesh_timescale, tau_adiost_define_mesh_timescale, "adios_define_mesh_timescale");
    CHECK(adiost_event_define_mesh_timeseriesformat, tau_adiost_define_mesh_timeseriesformat, "adios_define_mesh_timeseriesformat");
    CHECK(adiost_event_define_mesh_group, tau_adiost_define_mesh_group, "adios_define_mesh_group");
    CHECK(adiost_event_define_mesh_file, tau_adiost_define_mesh_file, "adios_define_mesh_file");
    CHECK(adiost_event_define_mesh_uniform, tau_adiost_define_mesh_uniform, "adios_define_mesh_uniform");
    CHECK(adiost_event_define_mesh_rectilinear, tau_adiost_define_mesh_rectilinear, "adios_define_mesh_rectilinear");
    CHECK(adiost_event_define_mesh_structured, tau_adiost_define_mesh_structured, "adios_define_mesh_structured");
    CHECK(adiost_event_define_mesh_unstructured, tau_adiost_define_mesh_unstructured, "adios_define_mesh_unstructured");
	/* ADIOS Read API events */
	CHECK(adiost_event_read_init_method, tau_adiost_read_init_method, "adios_read_init_method");
   	CHECK(adiost_event_read_finalize_method, tau_adiost_read_finalize_method, "adios_read_finalize_method");
   	CHECK(adiost_event_read_open, tau_adiost_read_open, "adios_read_open");
   	CHECK(adiost_event_read_open_file, tau_adiost_read_open_file, "adios_read_open_file");
   	CHECK(adiost_event_inq_var, tau_adiost_inq_var, "adios_inq_var");
   	CHECK(adiost_event_inq_var_byid, tau_adiost_inq_var_byid, "adios_inq_var_byid");
   	CHECK(adiost_event_free_varinfo, tau_adiost_free_varinfo, "adios_free_varinfo");
   	CHECK(adiost_event_inq_var_stat, tau_adiost_inq_var_stat, "adios_inq_var_stat");
   	CHECK(adiost_event_inq_var_blockinfo, tau_adiost_inq_var_blockinfo, "adios_inq_var_blockinfo");
   	CHECK(adiost_event_selection_boundingbox, tau_adiost_selection_boundingbox, "adios_selection_boundingbox");
   	CHECK(adiost_event_selection_points, tau_adiost_selection_points, "adios_selection_points");
   	CHECK(adiost_event_selection_writeblock, tau_adiost_selection_writeblock, "adios_selection_writeblock");
   	CHECK(adiost_event_selection_auto, tau_adiost_selection_auto, "adios_selection_auto");
   	CHECK(adiost_event_selection_delete, tau_adiost_selection_delete, "adios_selection_delete");
   	CHECK(adiost_event_schedule_read, tau_adiost_schedule_read, "adios_schedule_read");
   	CHECK(adiost_event_schedule_read_byid, tau_adiost_schedule_read_byid, "adios_schedule_read_byid");
   	CHECK(adiost_event_perform_reads, tau_adiost_perform_reads, "adios_perform_reads");
   	CHECK(adiost_event_check_reads, tau_adiost_check_reads, "adios_check_reads");
   	CHECK(adiost_event_free_chunk, tau_adiost_free_chunk, "adios_free_chunk");
   	CHECK(adiost_event_get_attr, tau_adiost_get_attr, "adios_get_attr");
   	CHECK(adiost_event_get_attr_byid, tau_adiost_get_attr_byid, "adios_get_attr_byid");
   	CHECK(adiost_event_type_to_string, tau_adiost_type_to_string, "adios_type_to_string");
   	CHECK(adiost_event_type_size, tau_adiost_type_size, "adios_type_size");
   	CHECK(adiost_event_get_grouplist, tau_adiost_get_grouplist, "adios_get_grouplist");
   	CHECK(adiost_event_group_view, tau_adiost_group_view, "adios_group_view");
   	CHECK(adiost_event_stat_cov, tau_adiost_stat_cov, "adios_stat_cov");
   	CHECK(adiost_event_inq_mesh_byid, tau_adiost_inq_mesh_byid, "adios_inq_mesh_byid");
   	CHECK(adiost_event_free_meshinfo, tau_adiost_free_meshinfo, "adios_free_meshinfo");
   	CHECK(adiost_event_inq_var_meshinfo, tau_adiost_inq_var_meshinfo, "adios_inq_var_meshinfo");
}

// Strong definition of ADIOS tool method
ADIOST_EXTERN ADIOST_EXPORT adiost_initialize_t adiost_tool() { return TAU_adiost_initialize; }

