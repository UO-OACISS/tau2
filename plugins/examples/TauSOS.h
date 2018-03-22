#ifndef TAU_SOS_H
#define TAU_SOS_H

#define TAU_SOS_INTERRUPT_PERIOD 2 // two seconds
#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.
#define TAU_SOS_DEFAULT 1 // if the plugin is loaded, use it!
#define TAU_SOS_TRACING_DEFAULT 0
#define TAU_SOS_TRACE_ADIOS_DEFAULT 0
#define TAU_SOS_PERIODIC_DEFAULT 0
#define TAU_SOS_PERIOD_DEFAULT 2000000 // microseconds


class SOS_plugin_options {
    private:
        SOS_plugin_options(void) :
            env_sos_enabled(TAU_SOS_DEFAULT),
            env_sos_tracing(TAU_SOS_TRACING_DEFAULT),
            env_sos_trace_adios(TAU_SOS_TRACE_ADIOS_DEFAULT),
            env_sos_periodic(TAU_SOS_PERIODIC_DEFAULT),
            env_sos_period(TAU_SOS_PERIOD_DEFAULT) {}
    public:
        int env_sos_enabled;
        int env_sos_tracing;
        int env_sos_trace_adios;
        int env_sos_periodic;
        int env_sos_period;
        static SOS_plugin_options& thePluginOptions() {
            static SOS_plugin_options tpo;
            return tpo;
        }
};

inline SOS_plugin_options& thePluginOptions() { 
    return SOS_plugin_options::thePluginOptions(); 
}

void TAU_SOS_parse_environment_variables(void);

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
