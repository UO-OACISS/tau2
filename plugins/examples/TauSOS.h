#ifndef TAU_SOS_H
#define TAU_SOS_H

#include <string>
#include <set>

#define TAU_SOS_INTERRUPT_PERIOD 2 // two seconds
#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.
#define TAU_SOS_DEFAULT 1 // if the plugin is loaded, use it!
#define TAU_SOS_TRACING_DEFAULT 0
#define TAU_SOS_TRACE_ADIOS_DEFAULT 0
#define TAU_SOS_PERIODIC_DEFAULT 0
#define TAU_SOS_PERIOD_DEFAULT 2000000 // microseconds
#define TAU_SOS_SHUTDOWN_DELAY_DEFAULT 10 // seconds
#define TAU_SOS_USE_SELECTION_DEFAULT 0 // microseconds
#define TAU_SOS_CACHE_DEPTH_DEFAULT 10 // SOS frames

class SOS_plugin_options {
    private:
        SOS_plugin_options(void) :
            env_sos_enabled(TAU_SOS_DEFAULT),
            env_sos_tracing(TAU_SOS_TRACING_DEFAULT),
            env_sos_trace_adios(TAU_SOS_TRACE_ADIOS_DEFAULT),
            env_sos_periodic(TAU_SOS_PERIODIC_DEFAULT),
            env_sos_period(TAU_SOS_PERIOD_DEFAULT),
            env_sos_shutdown_delay(TAU_SOS_SHUTDOWN_DELAY_DEFAULT),
            env_sos_use_selection(TAU_SOS_USE_SELECTION_DEFAULT),
            env_sos_cache_depth(TAU_SOS_CACHE_DEPTH_DEFAULT) {}
    public:
        int env_sos_enabled;
        int env_sos_tracing;
        int env_sos_trace_adios;
        int env_sos_periodic;
        int env_sos_period;
        int env_sos_shutdown_delay;
        int env_sos_use_selection;
        int env_sos_cache_depth;
        std::set<std::string> included_timers;
        std::set<std::string> excluded_timers;
        std::set<std::string> included_counters;
        std::set<std::string> excluded_counters;
        static SOS_plugin_options& thePluginOptions() {
            static SOS_plugin_options tpo;
            return tpo;
        }
};

inline SOS_plugin_options& thePluginOptions() { 
    return SOS_plugin_options::thePluginOptions(); 
}

void TAU_SOS_parse_environment_variables(void);
void Tau_SOS_parse_selection_file(const char * filename);
const bool Tau_SOS_contains(std::set<std::string>& myset, 
        const char * key, bool if_empty);

void TAU_SOS_send_data(void);
void TAU_SOS_init(void);
void TAU_SOS_stop_worker(void);
void TAU_SOS_finalize(void);
void TAU_SOS_send_data(void);
void Tau_SOS_pack_current_timer(const char * event_name);
void Tau_SOS_pack_string(const char * name, char * value);
void Tau_SOS_pack_double(const char * name, double value);
void Tau_SOS_pack_integer(const char * name, int value);
void Tau_SOS_pack_long(const char * name, long int value);
void * Tau_sos_thread_function(void* data);

#endif // TAU_SOS_H
