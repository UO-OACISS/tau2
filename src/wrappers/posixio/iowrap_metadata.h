#include <Profile/Profiler.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>

/* These functions are used in all three methods, so rather than copy the code
 * everywhere, put it in a header and include it. */

/* Helper function for generating metadata for linking workflow components */
void write_file_metadata(int tid, const char * parent_profiler, int flags,
        x_uint64 timestamp, const char * pathname) {
  /* write a metadata event for linking workflow components! */
  static int index = 0;
  char metadata_name[128] = {0};
  snprintf(metadata_name, sizeof(metadata_name),  "posix open[%d]", index);
  index = index + 1;
  char event_type[128] = {0};
  if (flags & O_WRONLY) {
      snprintf(event_type, sizeof(event_type),  "output");
  } else if (flags & O_RDWR) {
      snprintf(event_type, sizeof(event_type),  "input/output");
  } else { // O_RDONLY is 0, so it's our default.
      snprintf(event_type, sizeof(event_type),  "input");
  }
  char metadata_value[1024] = {0};
  snprintf(metadata_value, sizeof(metadata_value),  "{\"event-type\":\"%s\",\"name\":\"%s\",\"time\":\"%llu\",\"node-id\":\"%d\",\"thread-id\":\"%d\",\"filename\":\"%s\"}", event_type, parent_profiler, timestamp, Tau_get_node(), tid, pathname);
  TAU_METADATA(metadata_name, metadata_value);
  //printf("%s - %s\n", metadata_name, metadata_value);
}

#define TAU_IOWRAPPER_METADATA_SETUP \
  /* get the name of the current timer */ \
  int tau_io_tid = Tau_get_thread(); \
  const char * tau_io_parent = Tau_get_current_timer_name(tau_io_tid); \
  x_uint64 tau_io_stamp = TauMetrics_getTimeOfDay();

#define TAU_IOWRAPPER_WRITE_FILE_METADATA(tau_io_flags, tau_io_pathname) \
  write_file_metadata(tau_io_tid, tau_io_parent, tau_io_flags, tau_io_stamp, tau_io_pathname);

#define TAU_IOWRAPPER_WRITE_FILE_METADATA_FOPEN(tau_io_mode, tau_io_pathname) \
  int tau_io_flags = 0; \
  if (strstr(tau_io_mode, "r") != NULL) { tau_io_flags = O_RDONLY; } \
  else if (strstr(tau_io_mode, "w") != NULL) { tau_io_flags = O_WRONLY; } \
  else { tau_io_flags = O_RDWR; } \
  write_file_metadata(tau_io_tid, tau_io_parent, tau_io_flags, tau_io_stamp, tau_io_pathname);
