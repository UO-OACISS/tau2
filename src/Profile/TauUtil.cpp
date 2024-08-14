/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauUtil.cpp                                      **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : This file contains utility routines              **
**                                                                         **
****************************************************************************/

#include <TauUtil.h>
#include <TauPlugin.h>
#include <string>
#include <TauEnv.h>
#include <TauPluginInternals.h>
#include <TauPluginCPPTypes.h>
#include <stdarg.h>
#include <string.h>
#include <map>
#include <set>
#include <sstream>
#include <mutex>

#if defined TAU_USE_STDCXX11 || defined TAU_WINDOWS
#include <thread>
#include <regex>
#else
#include <regex.h>
#endif

#include <Profile/Profiler.h>
#include <TauMetaData.h>
#include <Profiler.h>

#ifdef HAVE_TR1_HASH_MAP
#include <tr1/functional>
#endif /* HAVE_TR1_HASH_MAP */

#ifndef TAU_WINDOWS
#include <dlfcn.h>
#else
#define strtok_r(a,b,c) strtok(a,b)
#endif /* TAU_WINDOWS */

/* Plugin Declarations */
std::map < PluginKey, std::set<unsigned int> >& Tau_get_plugins_for_named_specific_event(void) {
    static std::map < PluginKey, std::set<unsigned int> > my_map;
    return my_map;
}
std::map < unsigned int, Tau_plugin_new_t*>& Tau_get_plugin_map(void) {
    static std::map < unsigned int, Tau_plugin_new_t*> themap;
    return themap;
}
std::map < unsigned int, Tau_plugin_callbacks_t* >& Tau_get_plugin_callback_map(void) {
    static std::map < unsigned int, Tau_plugin_callbacks_t* > themap;
    return themap;
}

/* TODO: Temporary subtitute to "plugins_for_named_specific_event",
 * see TauPluginCPPTypes.h for more detail */
OmptPluginsVect plugins_for_ompt_event[NB_TAU_PLUGIN_EVENTS];

std::list < std::string > regex_list;

unsigned int plugin_id_counter = 0;
size_t star_hash;

extern "C" void Tau_enable_all_plugins_for_specific_event(int ev, const char *name);

Tau_plugin_callbacks_active_t Tau_plugins_enabled;

extern void Tau_ompt_register_plugin_callbacks(Tau_plugin_callbacks_active_t *Tau_plugins_enabled);
/* Plugin Declarations */


#define TAU_NAME_LENGTH 1024

#ifdef TAU_BFD
#include <Profile/TauBfd.h>

/*Data structures to return function context info*/
struct HashNode
{
  HashNode() : fi(NULL), excluded(false)
  { }

  TauBfdInfo info;		///< Filename, line number, etc.
  FunctionInfo * fi;		///< Function profile information
  char * resolved_name;   // save this for efficiency
  bool excluded;			///< Is function excluded from profiling?
};

struct HashTable : public TAU_HASH_MAP<unsigned long, HashNode*>
{
  HashTable() {
    Tau_init_initializeTAU();
  }
  virtual ~HashTable() {
    for (auto it = this->cbegin(); it != this->cend() /* not hoisted */; /* no increment */)
    {
        this->erase(it++);
    }
    Tau_destructor_trigger();
  }
};

std::mutex & TheHashMutex() {
    static std::mutex mtx;
    return mtx;
}

static HashTable & TheHashTable()
{
  static HashTable htab;
  return htab;
}

static tau_bfd_handle_t & TheBfdUnitHandle()
{
  static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    RtsLayer::LockEnv();
    if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
      bfdUnitHandle = Tau_bfd_registerUnit();
    }
    RtsLayer::UnLockEnv();
  }
  return bfdUnitHandle;
}
#endif /* TAU_BFD */

/* Given the function info object, resolve and return the address
 * that has been embedded in the function name using a pre-fixed token sequence.
 * Currently, this is only invoked from TracerOTF2.cpp and Profiler.cpp while
 * writing out the trace and profile files respectively.
 * NOTE: We do NOT need to lock the HashTable data structure as the thread has already
 * acquired the lock from outside this routine */
extern "C" void Tau_ompt_resolve_callsite(FunctionInfo &fi, char * resolved_address) {

      unsigned long addr = 0;
      char region_type[100];
      sscanf(fi.GetName(), "%s ADDR <%lx>", region_type, &addr);
#ifdef TAU_BFD
      // my local map, to reduce contention
      thread_local static TAU_HASH_MAP<unsigned long, HashNode*> local_map;
      HashNode * node;
      node = local_map[addr];
      // not in my local map?  look in the global map
      if (!node) {
        // acquire lock for global map
        std::lock_guard<std::mutex> lck (TheHashMutex());
        node = TheHashTable()[addr];
        if (!node) {
            node = new HashNode;
            node->fi = NULL;
            node->excluded = false;
            TheHashTable()[addr] = node;
        }
        local_map[addr] = node;
      }

      // resolve the string for output
      tau_bfd_handle_t & bfdUnitHandle = TheBfdUnitHandle();
      Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, node->info);

      if(node && node->info.filename && node->info.funcname && node->info.lineno) {
        sprintf(resolved_address, "%s %s [{%s} {%d, 0}]", region_type, node->info.funcname, node->info.filename, node->info.lineno);
      } else if(node && node->info.filename && node->info.funcname) {
        sprintf(resolved_address, "%s %s [{%s} {0, 0}]", region_type, node->info.funcname, node->info.filename);
      } else if(node && node->info.funcname) {
        sprintf(resolved_address, "%s %s", region_type, node->info.funcname);
      } else {
        sprintf(resolved_address, "%s __UNKNOWN__", region_type);
      }
#else
        sprintf(resolved_address, "%s __UNKNOWN__", region_type);
#endif /*TAU_BFD*/
}

/* Given the unsigned long address, and a pointer to the string, fill the string with the BFD resolved address.
 * NOTE: We need to lock the HashTable data structure, as this function is invoked from the OMPT callbacks themselves,
 * when the user wants to resolve the function name eagerly.
 * For this feature to be active, TAU_OMPT_RESOLVE_ADDRESS_EAGERLY must be set.*/
extern "C" void Tau_ompt_resolve_callsite_eagerly(unsigned long addr, char * resolved_address) {

      #ifdef TAU_BFD
      HashNode * node;
      tau_bfd_handle_t & bfdUnitHandle = TheBfdUnitHandle();

      // my local map, to reduce contention
      thread_local static TAU_HASH_MAP<unsigned long, HashNode*> local_map;
      node = local_map[addr];
      // not in my local map?  look in the global map
      if (!node) {
        // acquire lock for global map
        std::lock_guard<std::mutex> lck (TheHashMutex());
        node = TheHashTable()[addr];
        if (!node) {
            node = new HashNode;
            node->fi = NULL;
            node->excluded = false;
            TheHashTable()[addr] = node;
            Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, node->info);
            int length = strlen(node->info.funcname) + strlen(node->info.filename) + 64;
            node->resolved_name = (char*)(malloc(length * sizeof(char)));
            if(node && node->info.filename && node->info.funcname && node->info.lineno) {
                sprintf(node->resolved_name, "%s [{%s} {%d, 0}]", node->info.funcname, node->info.filename, node->info.lineno);
            } else if(node && node->info.filename && node->info.funcname) {
                sprintf(node->resolved_name, "%s [{%s} {0, 0}]", node->info.funcname, node->info.filename);
            } else if(node && node->info.funcname) {
                sprintf(node->resolved_name, "%s", node->info.funcname);
            } else {
                sprintf(node->resolved_name, "__UNKNOWN__");
            }
        }
        local_map[addr] = node;
      }
      sprintf(resolved_address, "%s", node->resolved_name);

      #else
        sprintf(resolved_address, "__UNKNOWN__");
      #endif /*TAU_BFD*/
}

extern "C" size_t Tau_util_return_hash_of_string(const char * input) {
#if defined(TAU_USE_STDCXX11) || (defined(__clang__) || defined (TAU_FX_AARCH64) ) // && defined(__APPLE__)
  std::hash<std::string> hash_fn;
#else
  std::tr1::hash<std::string> hash_fn;
#endif
  std::string s(input);
  return hash_fn(s);
}

/*********************************************************************
 * Abort execution with a message
 ********************************************************************/
void TAU_ABORT(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  exit(EXIT_FAILURE);
}



/*********************************************************************
 * Create an buffer output device
 ********************************************************************/
Tau_util_outputDevice *Tau_util_createBufferOutputDevice()
{
  Tau_util_outputDevice *out = (Tau_util_outputDevice*) TAU_UTIL_MALLOC (sizeof(Tau_util_outputDevice));
  if (out == NULL) {
    return NULL;
  }
  out->type = TAU_UTIL_OUTPUT_BUFFER;
  out->bufidx = 0;
  out->buflen = TAU_UTIL_INITIAL_BUFFER;
  out->buffer = (char *)malloc(out->buflen + 1);
  return out;
}

/*********************************************************************
 * Return output buffer
 ********************************************************************/
char *Tau_util_getOutputBuffer(Tau_util_outputDevice *out) {
  return out->buffer;
}

/*********************************************************************
 * Return output buffer length
 ********************************************************************/
int Tau_util_getOutputBufferLength(Tau_util_outputDevice *out) {
  return out->bufidx;
}

/*********************************************************************
 * Free and close output device
 ********************************************************************/
void Tau_util_destroyOutputDevice(Tau_util_outputDevice *out) {
  if (out->type == TAU_UTIL_OUTPUT_BUFFER) {
    free (out->buffer);
  } else {
    fclose(out->fp);
  }
  free (out);
}

/*********************************************************************
 * Write to output device
 ********************************************************************/
int Tau_util_output(Tau_util_outputDevice *out, const char *format, ...) {
  int rs;
  va_list args;
  if (out->type == TAU_UTIL_OUTPUT_BUFFER) {
    va_start(args, format);
    rs = vsprintf(out->buffer+out->bufidx, format, args);
    va_end(args);
    out->bufidx+=rs;
    if (out->bufidx+TAU_UTIL_OUTPUT_THRESHOLD > out->buflen) {
      out->buflen = out->buflen * 2;
      out->buffer = (char*) realloc (out->buffer, out->buflen);
    }
  } else {
    va_start(args, format);
    rs = vfprintf(out->fp, format, args);
    va_end(args);
  }
  return rs;
}

/*********************************************************************
 * Read an entire line from a file
 ********************************************************************/
int Tau_util_readFullLine(char *line, FILE *fp) {
  int ch;
  int i = 0;
  while ( (ch = fgetc(fp)) && ch != EOF && ch != (int) '\n') {
    line[i++] = (unsigned char) ch;
  }
  // Be careful to check that line is large enough:
  // sizeof(line) == strlen(str) + 1
  line[i] = '\0';
  return i;
}

/*********************************************************************
 * Duplicates a string and replaces all the runs of spaces with a
 * single space.
 ********************************************************************/
char const * Tau_util_removeRuns(char const * spaced_str)
{
  if (!spaced_str) {
    return spaced_str; /* do nothing with a null string */
  }

  // Skip over spaces at start of string
  while (*spaced_str && *spaced_str == ' ') {
    ++spaced_str;
  }

  // String copy
  int len = strlen(spaced_str);
  char * str = (char *)malloc(len+1);

  // Copy from spaced_str ignoring runs of multiple spaces
  char c;
  char * dst = str;
  char const * src = spaced_str;
  char const * end = spaced_str + len;
  while ((c = *src) && src < end) {
    ++src;
    *dst = c;
    ++dst;
    if(c == ' ')
      while(*src == ' ')
        ++src;
  }
  *dst = '\0';

  return str;
}


void *Tau_util_malloc(size_t size, const char *file, int line) {
  void *ptr = malloc (size);
  if (!ptr) {
    TAU_ABORT("TAU: Abort: Unable to allocate memory (malloc) at %s:%d\n", file, line);
  }
  return ptr;
}

void *Tau_util_calloc(size_t size, const char *file, int line) {
  void *ptr = calloc (1,size);
  if (!ptr) {
    TAU_ABORT("TAU: Abort: Unable to allocate memory (calloc) at %s:%d\n", file, line);
  }
  return ptr;
}

/*********************************************************************
 * Create and return a new plugin manager if plugin system is un-initialized.
 * If it is already initialized, return a reference to the same plugin manager - Singleton Pattern
 ********************************************************************/
PluginManager* Tau_util_get_plugin_manager() {
  static PluginManager * plugin_manager = NULL;
  static int is_plugin_system_initialized = 0;

  /*Allocate memory for the plugin list and callback list*/
  if(!is_plugin_system_initialized) {

    plugin_manager = (PluginManager*)malloc(sizeof(PluginManager));
    plugin_manager->plugin_list = (Tau_plugin_list *)malloc(sizeof(Tau_plugin_list));
    (plugin_manager->plugin_list)->head = NULL;

    plugin_manager->callback_list = (Tau_plugin_callback_list*)malloc(sizeof(Tau_plugin_callback_list));
    (plugin_manager->callback_list)->head = NULL;
    is_plugin_system_initialized = 1;
  }

  return plugin_manager;
}

/*********************************************************************
 * Initializes the plugin system by loading and registering all plugins
 ********************************************************************/
int Tau_initialize_plugin_system() {
  memset(&Tau_plugins_enabled, 0, sizeof(Tau_plugin_callbacks_active_t));

  if(TauEnv_get_plugins_enabled()) {
    TAU_VERBOSE("TAU INIT: Initializing plugin system...\n");
    if(!Tau_util_load_and_register_plugins(Tau_util_get_plugin_manager())) {
      TAU_VERBOSE("TAU INIT: Successfully Initialized the plugin system.\n");
    } else {
      printf("TAU INIT: Error initializing the plugin system\n");
    }
  }
  return 0;
}

/*********************************************************************
 * Internal function that helps parse a token for the plugin name
 ********************************************************************/
int Tau_util_parse_plugin_token(char * token, char ** plugin_name, char *** plugin_args, int * plugin_num_args) {
  int length_of_arg_string = -1;
  char * save_ptr;
  char * arg_string;
  char * arg_token;
  char *pos_left = NULL;
  char *pos_right = NULL;


  *plugin_num_args = 0;
  *plugin_name = (char*)calloc(1024,sizeof(char));
  pos_left = strchr(token, '(');
  pos_right = strchr(token, ')');

  if(pos_left == NULL && pos_right == NULL) {
    strncpy(*plugin_name,  token, 1024); 
    return 0;
  } else if (pos_left == NULL || pos_right == NULL) {
    return -1; //Bad plugin name
  }

  *plugin_args = (char**)calloc(10,sizeof(char*)); //Maximum of 10 args supported for now
  arg_string = (char*)calloc(1024,sizeof(char));

  length_of_arg_string = (pos_right - pos_left) - 1;

  strncpy(arg_string, pos_left+1, length_of_arg_string);
  // null terminate the string after copying it.
  arg_string[length_of_arg_string] = '\0';
  strncpy(*plugin_name, token, (pos_left-token));

  arg_token = strtok_r(arg_string, ",", &save_ptr);

  int i = 0;
  /*Grab and pack, and count all the arguments to the plugin*/
  while(arg_token != NULL) {
    (*plugin_num_args)++;
    (*plugin_args)[i] = (char*)calloc(1024,sizeof(char));
    strncpy((*plugin_args)[i],  arg_token, 1024); 
    arg_token = strtok_r(NULL, ",", &save_ptr);
    i++;
  }

  TAU_VERBOSE("TAU PLUGIN: Arg string and count for token %s are %s and %d\n", token, arg_string, *plugin_num_args);

  return 0;
}

void Tau_enable_plugins_for_all_events() {
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_METADATA_REGISTRATION, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_POST_INIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_DUMP, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_MPIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_FUNCTION_ENTRY, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_FUNCTION_EXIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_PHASE_ENTRY, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_PHASE_EXIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_SEND, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_RECV, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_ATOMIC_EVENT_REGISTRATION, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_END_OF_EXECUTION, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_FUNCTION_FINALIZE, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_INTERRUPT_TRIGGER, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_TRIGGER, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_PARALLEL_BEGIN, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_PARALLEL_END, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_TASK_CREATE, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_TASK_SCHEDULE, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_IMPLICIT_TASK, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_THREAD_BEGIN, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_THREAD_END, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_WORK, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_MASTER, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_IDLE, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_SYNC_REGION, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRE, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRED, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_MUTEX_RELEASED, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_TARGET, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_TARGET_DATA_OP, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_TARGET_SUBMIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_OMPT_FINALIZE, "*");
  /* GPU EVENTS BEGIN */
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_GPU_INIT, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_GPU_FINALIZE, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_GPU_KERNEL_EXEC, "*");
  Tau_enable_all_plugins_for_specific_event(TAU_PLUGIN_EVENT_GPU_MEMCPY, "*");
  /* GPU EVENTS END */
}

/* TODO: Function part of the tomporary fix described in
 * TauPluginCPPTypes.h */
void Tau_flag_ompt_events() {
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_PARALLEL_BEGIN].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_PARALLEL_END].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_TASK_CREATE].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_TASK_SCHEDULE].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_IMPLICIT_TASK].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_THREAD_BEGIN].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_THREAD_END].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_WORK].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_MASTER].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_IDLE].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_SYNC_REGION].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRE].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRED].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_MUTEX_RELEASED].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_TARGET].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_TARGET_DATA_OP].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_TARGET_SUBMIT].flag_as_ompt();
  plugins_for_ompt_event[TAU_PLUGIN_EVENT_OMPT_FINALIZE].flag_as_ompt();
}


/*********************************************************************
 * Load a list of plugins at TAU init, given following environment variables:
 *  - TAU_PLUGINS_NAMES
 *  - TAU_PLUGINS_PATH
********************************************************************* */
int Tau_util_load_and_register_plugins(PluginManager* plugin_manager)
{
  char pluginpath[1024];
  char listpluginsnames[1024];
  char *token = NULL;
  char *plugin_name = NULL;
  //char *initFuncName = NULL;
  char **plugin_args;
  char *save_ptr;
  int plugin_num_args;

  if((TauEnv_get_plugins_path() == NULL) || (TauEnv_get_plugins() == NULL)) {
    printf("TAU: One or more of the environment variable(s) TAU_PLUGINS_PATH: %s, TAU_PLUGINS: %s are empty\n", TauEnv_get_plugins_path(), TauEnv_get_plugins());
    return -1;
  }

  strncpy(pluginpath,  TauEnv_get_plugins_path(), sizeof(pluginpath)); 
  strncpy(listpluginsnames,  TauEnv_get_plugins(), sizeof(listpluginsnames)); 

  /*Individual plugin names are separated by a ":"*/
  token = strtok_r(listpluginsnames,":", &save_ptr);
  TAU_VERBOSE("TAU: Trying to load plugin with name %s\n", token);

  std::set<std::string> plugins_seen;

  while(token != NULL)
  {
    // check to make sure we haven't loaded it already!
    std::string tmp(token);
    if (plugins_seen.count(tmp) > 0) {
        token = strtok_r(NULL, ":", &save_ptr);
        continue;
    }
    TAU_VERBOSE("TAU: Loading plugin: %s\n", token);
    if (Tau_util_parse_plugin_token(token, &plugin_name, &plugin_args, &plugin_num_args)) {
      printf("TAU: Plugin name specification does not match form <plugin_name1>(<plugin_arg1>,<plugin_arg2>):<plugin_name2>(<plugin_arg1>,<plugin_arg2>) for: %s\n",token);
      return -1;
    }
    plugins_seen.insert(tmp);

    std::stringstream ss;
#ifndef TAU_WINDOWS
    ss << pluginpath << "/" << plugin_name;
#else
    ss << pluginpath << "\\" << plugin_name;
#endif
    std::string fullpath{ss.str()};

    TAU_VERBOSE("TAU: Full path for the current plugin: %s\n", fullpath.c_str());

    /*Return a handle to the loaded dynamic object*/
    void* handle = Tau_util_load_plugin(plugin_name, fullpath.c_str(), plugin_manager);

    if (handle) {
      /*If handle is NOT NULL, register the plugin's handlers for various supported events*/
      handle = Tau_util_register_plugin(plugin_name, plugin_args, plugin_num_args, handle, plugin_manager, plugin_id_counter);

      /*Plugin registration failed. Bail*/
      if(!handle) return -1;
      TAU_VERBOSE("TAU: Successfully called the init func of plugin: %s\n", token);

      /* Plugin API */
      Tau_plugin_new_t * plugin_;
      plugin_ = (Tau_plugin_new_t *)malloc(sizeof(Tau_plugin_new_t));

      strcpy(plugin_->plugin_name, plugin_name);
      plugin_->id = plugin_id_counter;
      plugin_->handle = handle;
      Tau_get_plugin_map()[plugin_id_counter] = plugin_;
      plugin_id_counter++;
      /* Plugin API */

    } else {
      /*Plugin loading failed for some reason*/
      return -1;
    }

    token = strtok_r(NULL, ":", &save_ptr);
  }

  Tau_flag_ompt_events();
  Tau_enable_plugins_for_all_events();
  star_hash = Tau_util_return_hash_of_string("*");

  Tau_metadata_push_to_plugins();

  return 0;
}

/**************************************************************************************************************************
 * Use dlsym to find a function : TAU_PLUGIN_INIT_FUNC that the plugin MUST implement in order to register itself.
 * If plugin registration succeeds, then the callbacks for that plugin have been added to the plugin manager's callback list
 * ************************************************************************************************************************/
void* Tau_util_register_plugin(const char *name, char **args, int num_args, void* handle, PluginManager* plugin_manager, unsigned int plugin_id) {
#ifndef TAU_WINDOWS
  PluginInitFunc init_func = (PluginInitFunc) dlsym(handle, TAU_PLUGIN_INIT_FUNC);
#else
  PluginInitFunc init_func = (PluginInitFunc) NULL;
#endif /* TAU_WINDOWS */

  if(!init_func) {
#ifndef TAU_WINDOWS
    printf("TAU: Failed to retrieve TAU_PLUGIN_INIT_FUNC from plugin %s with error:%s\n", name, dlerror());
    dlclose(handle); //TODO : Replace with Tau_plugin_cleanup();
#endif /* TAU_WINDOWS */
    return NULL;
  }

  int return_val = init_func(num_args, args, plugin_id);
  if(return_val < 0) {
    printf("TAU: Call to init func for plugin %s returned failure error code %d\n", name, return_val);
#ifndef TAU_WINDOWS
    dlclose(handle); //TODO : Replace with Tau_plugin_cleanup();
#endif /* TAU_WINDOWS */
    return NULL;
  }
  return handle;
}

/**************************************************************************************************************************
 * Given a plugin name and fullpath, load a plugin and return a handle to the opened DSO
 * ************************************************************************************************************************/
void* Tau_util_load_plugin(const char *name, const char *path, PluginManager* plugin_manager) {
#ifndef TAU_WINDOWS
  void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
#else
  void* handle = NULL;
#endif /* TAU_WINDOWS */

  if (handle) {
    Tau_plugin * plugin = (Tau_plugin *)malloc(sizeof(Tau_plugin));
    strcpy(plugin->plugin_name, name);
    plugin->handle = handle;
    plugin->next = (plugin_manager->plugin_list)->head;
    (plugin_manager->plugin_list)->head = plugin;

    TAU_VERBOSE("TAU: Successfully loaded plugin: %s\n", name);
    return handle;
  } else {
#ifndef TAU_WINDOWS
    printf("TAU: Failed loading %s plugin with error: %s\n", name, dlerror());
#endif /* TAU_WINDOWS */
    return NULL;
  }
}

/**************************************************************************************************************************
 * Initialize Tau_plugin_callbacks structure with default values
 * This is necessary in order to prevent future event additions to affect older plugins
 * ************************************************************************************************************************/
extern "C" void Tau_util_init_tau_plugin_callbacks(Tau_plugin_callbacks * cb) {
  cb->FunctionRegistrationComplete = 0;
  cb->MetadataRegistrationComplete = 0;
  cb->PostInit = 0;
  cb->Dump = 0;
  cb->Mpit = 0;
  cb->FunctionEntry = 0;
  cb->FunctionExit = 0;
  cb->Send = 0;
  cb->Recv = 0;
  cb->CurrentTimerExit = 0;
  cb->AtomicEventRegistrationComplete = 0;
  cb->AtomicEventTrigger = 0;
  cb->PreEndOfExecution = 0;
  cb->EndOfExecution = 0;
  cb->InterruptTrigger = 0;
  cb->Trigger = 0;
  cb->FunctionFinalize = 0;
  cb->PhaseEntry = 0;
  cb->PhaseExit = 0;
  cb->OmptParallelBegin = 0;
  cb->OmptParallelEnd = 0;
  cb->OmptTaskCreate = 0;
  cb->OmptTaskSchedule = 0;
  cb->OmptImplicitTask = 0;
  cb->OmptThreadBegin = 0;
  cb->OmptThreadEnd = 0;
  cb->OmptWork = 0;
  cb->OmptMaster = 0;
  cb->OmptIdle = 0;
  cb->OmptSyncRegion = 0;
  cb->OmptMutexAcquire = 0;
  cb->OmptMutexAcquired = 0;
  cb->OmptMutexReleased = 0;
  cb->OmptTarget = 0;
  cb->OmptTargetDataOp = 0;
  cb->OmptTargetSubmit = 0;
  cb->OmptFinalize = 0;
  cb->GpuInit = 0;
  cb->GpuFinalize = 0;
  cb->GpuKernelExec = 0;
  cb->GpuMemcpy = 0;
}

/**************************************************************************************************************************
 * Helper function that makes a copy of all callbacks for events
 ***************************************************************************************************************************/
void Tau_util_make_callback_copy(Tau_plugin_callbacks * dest, Tau_plugin_callbacks * src) {
  dest->FunctionRegistrationComplete = src->FunctionRegistrationComplete;
  dest->MetadataRegistrationComplete = src->MetadataRegistrationComplete;
  dest->PostInit = src->PostInit;
  dest->Dump = src->Dump;
  dest->Mpit = src->Mpit;
  dest->FunctionEntry = src->FunctionEntry;
  dest->FunctionExit = src->FunctionExit;
  dest->Send = src->Send;
  dest->Recv = src->Recv;
  dest->CurrentTimerExit = src->CurrentTimerExit;
  dest->AtomicEventTrigger = src->AtomicEventTrigger;
  dest->AtomicEventRegistrationComplete = src->AtomicEventRegistrationComplete;
  dest->PreEndOfExecution = src->PreEndOfExecution;
  dest->EndOfExecution = src->EndOfExecution;
  dest->InterruptTrigger = src->InterruptTrigger;
  dest->Trigger = src->Trigger;
  dest->FunctionFinalize = src->FunctionFinalize;
  dest->PhaseEntry = src->PhaseEntry;
  dest->PhaseExit = src->PhaseExit;
  dest->OmptParallelBegin = src->OmptParallelBegin;
  dest->OmptParallelEnd = src->OmptParallelEnd;
  dest->OmptTaskCreate = src->OmptTaskCreate;
  dest->OmptTaskSchedule = src->OmptTaskSchedule;
  dest->OmptImplicitTask = src->OmptImplicitTask;
  dest->OmptThreadBegin = src->OmptThreadBegin;
  dest->OmptThreadEnd = src->OmptThreadEnd;
  dest->OmptWork = src->OmptWork;
  dest->OmptMaster = src->OmptMaster;
  dest->OmptIdle = src->OmptIdle;
  dest->OmptSyncRegion = src->OmptSyncRegion;
  dest->OmptMutexAcquire = src->OmptMutexAcquire;
  dest->OmptMutexAcquired = src->OmptMutexAcquired;
  dest->OmptMutexReleased = src->OmptMutexReleased;
  dest->OmptTarget = src->OmptTarget;
  dest->OmptTargetDataOp = src->OmptTargetDataOp;
  dest->OmptTargetSubmit = src->OmptTargetSubmit;
  dest->OmptFinalize = src->OmptFinalize;
  dest->GpuInit = src->GpuInit;
  dest->GpuFinalize = src->GpuFinalize;
  dest->GpuKernelExec = src->GpuKernelExec;
  dest->GpuMemcpy = src->GpuMemcpy;
}

/**************************************************************************************************************************
 * Register callbacks associated with well defined events defined in struct Tau_plugin_callbacks
 **************************************************************************************************************************/
extern "C" void Tau_util_plugin_register_callbacks(Tau_plugin_callbacks * cb, unsigned int plugin_id) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();

  Tau_plugin_callback_t * callback = (Tau_plugin_callback_t *)malloc(sizeof(Tau_plugin_callback_t));
  Tau_util_make_callback_copy(&(callback->cb), cb);
  callback->next = (plugin_manager->callback_list)->head;
  (plugin_manager->callback_list)->head = callback;

  /* Plugin API */
  Tau_plugin_callbacks_t * cb_ = (Tau_plugin_callbacks_t *)malloc(sizeof(Tau_plugin_callbacks_t));
  Tau_util_make_callback_copy(cb_, cb);
  Tau_get_plugin_callback_map()[plugin_id] = cb_;
  /* Plugin API */

  /* Set some flags to make runtime conditional processing more efficient */
  if (cb->FunctionRegistrationComplete != 0) { Tau_plugins_enabled.function_registration = 1; }
  if (cb->MetadataRegistrationComplete != 0) { Tau_plugins_enabled.metadata_registration = 1; }
  if (cb->PostInit != 0) { Tau_plugins_enabled.post_init = 1; }
  if (cb->Dump != 0) { Tau_plugins_enabled.dump = 1; }
  if (cb->Mpit != 0) { Tau_plugins_enabled.mpit = 1; }
  if (cb->FunctionEntry != 0) { Tau_plugins_enabled.function_entry = 1; }
  if (cb->FunctionExit != 0) { Tau_plugins_enabled.function_exit = 1; }
  if (cb->Send != 0) { Tau_plugins_enabled.send = 1; }
  if (cb->Recv != 0) { Tau_plugins_enabled.recv = 1; }
  if (cb->CurrentTimerExit != 0) { Tau_plugins_enabled.current_timer_exit = 1; }
  if (cb->AtomicEventRegistrationComplete != 0) { Tau_plugins_enabled.atomic_event_registration = 1; }
  if (cb->AtomicEventTrigger != 0) { Tau_plugins_enabled.atomic_event_trigger = 1; }
  if (cb->PreEndOfExecution != 0) { Tau_plugins_enabled.pre_end_of_execution = 1; }
  if (cb->EndOfExecution != 0) { Tau_plugins_enabled.end_of_execution = 1; }
  if (cb->FunctionFinalize != 0) { Tau_plugins_enabled.function_finalize = 1; }
  if (cb->InterruptTrigger != 0) { Tau_plugins_enabled.interrupt_trigger = 1; }
  if (cb->Trigger != 0) { Tau_plugins_enabled.trigger = 1; }
  if (cb->PhaseEntry != 0) { Tau_plugins_enabled.phase_entry = 1; }
  if (cb->PhaseExit != 0) { Tau_plugins_enabled.phase_exit = 1; }
  if (cb->OmptParallelBegin != 0) { Tau_plugins_enabled.ompt_parallel_begin = 1; }
  if (cb->OmptParallelEnd != 0) { Tau_plugins_enabled.ompt_parallel_end = 1; }
  if (cb->OmptTaskCreate != 0) { Tau_plugins_enabled.ompt_task_create = 1; }
  if (cb->OmptTaskSchedule != 0) { Tau_plugins_enabled.ompt_task_schedule = 1; }
  if (cb->OmptImplicitTask != 0) { Tau_plugins_enabled.ompt_implicit_task = 1; }
  if (cb->OmptThreadBegin != 0) { Tau_plugins_enabled.ompt_thread_begin = 1; }
  if (cb->OmptThreadEnd != 0) { Tau_plugins_enabled.ompt_thread_end = 1; }
  if (cb->OmptWork != 0) { Tau_plugins_enabled.ompt_work = 1; }
  if (cb->OmptMaster != 0) { Tau_plugins_enabled.ompt_master = 1; }
  if (cb->OmptIdle != 0) { Tau_plugins_enabled.ompt_idle = 1; }
  if (cb->OmptSyncRegion != 0) { Tau_plugins_enabled.ompt_sync_region = 1; }
  if (cb->OmptMutexAcquire != 0) { Tau_plugins_enabled.ompt_mutex_acquire = 1; }
  if (cb->OmptMutexAcquired != 0) { Tau_plugins_enabled.ompt_mutex_acquired = 1; }
  if (cb->OmptMutexReleased != 0) { Tau_plugins_enabled.ompt_mutex_released = 1; }
  if (cb->OmptTarget != 0) { Tau_plugins_enabled.ompt_target = 1; }
  if (cb->OmptTargetDataOp != 0) { Tau_plugins_enabled.ompt_target_data_op = 1; }
  if (cb->OmptTargetSubmit != 0) { Tau_plugins_enabled.ompt_target_submit = 1; }
  if (cb->OmptFinalize != 0) { Tau_plugins_enabled.ompt_finalize = 1; }
  if (cb->GpuInit != 0) { Tau_plugins_enabled.gpu_init = 1; }
  if (cb->GpuFinalize != 0) { Tau_plugins_enabled.gpu_finalize = 1; }
  if (cb->GpuKernelExec != 0) { Tau_plugins_enabled.gpu_kernel_exec = 1; }
  if (cb->GpuMemcpy != 0) { Tau_plugins_enabled.gpu_memcpy = 1; }

  /* Register needed OMPT callback if they are not already registered */
#if defined(TAU_USE_OMPT) || defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
  Tau_ompt_register_plugin_callbacks(&Tau_plugins_enabled);
#endif /* TAU_OMPT */
}

#if not defined TAU_USE_STDCXX11 && not defined TAU_WINDOWS
/* C version of regex_match in case compiler doesn't support C++11 featues */
/* Credit for logic: Laurence Gonsalves on stackoverflow.com */
extern "C" int Tau_C_regex_match(const char * input, const char * rege)
{
  TauInternalFunctionGuard protects_this_function;

  regex_t regex;
  int reti;
  char msgbuf[100];

  /* Compile regular expression */
  reti = regcomp(&regex, rege, 0);
  if (reti) {
      fprintf(stderr, "Could not compile regex\n");
      return 0;
  }

  /* Execute regular expression */
  reti = regexec(&regex, input, 0, NULL, 0);
  if (!reti) {
      return 1;
  }

  else {
      return 0;
  }
}
#endif

extern "C" const char* Tau_check_for_matching_regex(const char * input)
{

  TauInternalFunctionGuard protects_this_function;
#if defined TAU_USE_STDCXX11 || defined TAU_WINDOWS
  for(std::list< std::string >::iterator it = regex_list.begin(); it != regex_list.end(); it++) {
    if(regex_match(input, std::regex(*it))) {
      return (*it).c_str();
    }
  }
#else
  for(std::list< std::string >::iterator it = regex_list.begin(); it != regex_list.end(); it++) {
    if(Tau_C_regex_match(input, (*it).c_str())) {
      return (*it).c_str();
    }
  }
#endif
  return NULL;
}

/* TODO: for the following overloaded functions, the ones for the ompt events are slightly different due to a bug. More information in TauPluginCPPTypes.h */

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the function registration event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_registration_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->FunctionRegistrationComplete != 0)
      Tau_get_plugin_callback_map()[*it]->FunctionRegistrationComplete(data);
  }

}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the mpit event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_mpit_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->Mpit != 0)
      Tau_get_plugin_callback_map()[*it]->Mpit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the dump event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_dump_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->Dump != 0)
      Tau_get_plugin_callback_map()[*it]->Dump(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the function entry event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_entry_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->FunctionEntry != 0)
      Tau_get_plugin_callback_map()[*it]->FunctionEntry(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the function exit event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_exit_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->FunctionExit != 0)
      Tau_get_plugin_callback_map()[*it]->FunctionExit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the "current timer" exit event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_current_timer_exit_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->CurrentTimerExit != 0)
      Tau_get_plugin_callback_map()[*it]->CurrentTimerExit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the send event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_send_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->Send != 0)
      Tau_get_plugin_callback_map()[*it]->Send(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the recv event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_recv_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->Recv != 0)
      Tau_get_plugin_callback_map()[*it]->Recv(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the metadata registration event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_metadata_registration_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->MetadataRegistrationComplete != 0)
      Tau_get_plugin_callback_map()[*it]->MetadataRegistrationComplete(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the post init event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_post_init_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->PostInit != 0)
      Tau_get_plugin_callback_map()[*it]->PostInit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the atomic event registration event
 ****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_atomic_event_registration_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->AtomicEventRegistrationComplete != 0)
      Tau_get_plugin_callback_map()[*it]->AtomicEventRegistrationComplete(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the atomic event trigger event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_atomic_event_trigger_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->AtomicEventTrigger != 0)
      Tau_get_plugin_callback_map()[*it]->AtomicEventTrigger(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_parallel_begin event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_parallel_begin_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptParallelBegin != 0)
      Tau_get_plugin_callback_map()[id]->OmptParallelBegin(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_parallel_end event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_parallel_end_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptParallelEnd != 0)
      Tau_get_plugin_callback_map()[id]->OmptParallelEnd(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_task_create event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_task_create_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptTaskCreate != 0)
      Tau_get_plugin_callback_map()[id]->OmptTaskCreate(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_task_schedule event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_task_schedule_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptTaskSchedule != 0)
      Tau_get_plugin_callback_map()[id]->OmptTaskSchedule(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_implicit_task event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_implicit_task_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptImplicitTask != 0)
      Tau_get_plugin_callback_map()[id]->OmptImplicitTask(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_thread_begin event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_thread_begin_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptThreadBegin != 0)
      Tau_get_plugin_callback_map()[id]->OmptThreadBegin(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_thread_end event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_thread_end_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptThreadEnd != 0)
      Tau_get_plugin_callback_map()[id]->OmptThreadEnd(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_work event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_work_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptWork != 0)
      Tau_get_plugin_callback_map()[id]->OmptWork(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_master event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_master_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptMaster != 0)
      Tau_get_plugin_callback_map()[id]->OmptMaster(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_idle event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_idle_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptIdle != 0)
      Tau_get_plugin_callback_map()[id]->OmptIdle(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_sync_region event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_sync_region_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptSyncRegion != 0)
      Tau_get_plugin_callback_map()[id]->OmptSyncRegion(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_mutex_acquire event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_mutex_acquire_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptMutexAcquire != 0)
      Tau_get_plugin_callback_map()[id]->OmptMutexAcquire(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_mutex_acquired event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_mutex_acquired_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptMutexAcquired != 0)
      Tau_get_plugin_callback_map()[id]->OmptMutexAcquired(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_mutex_released event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_mutex_released_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptMutexReleased != 0)
      Tau_get_plugin_callback_map()[id]->OmptMutexReleased(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the pre end of execution event
 ******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_pre_end_of_execution_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->PreEndOfExecution != 0)
      Tau_get_plugin_callback_map()[*it]->PreEndOfExecution(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the end of execution event
 ******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_end_of_execution_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->EndOfExecution != 0)
      Tau_get_plugin_callback_map()[*it]->EndOfExecution(data);
  }
}

/*****************************************************************************
 * Overloaded function that invokes all registered callbacks for the
 * finalize event
 *****************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_finalize_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->FunctionFinalize != 0)
      Tau_get_plugin_callback_map()[*it]->FunctionFinalize(data);
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for interrupt trigger event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_interrupt_trigger_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->InterruptTrigger != 0)
      Tau_get_plugin_callback_map()[*it]->InterruptTrigger(data);
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for trigger event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_trigger_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->Trigger != 0) {
      Tau_get_plugin_callback_map()[*it]->Trigger(data);
    }
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for phase entry event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_phase_entry_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->PhaseEntry != 0)
      Tau_get_plugin_callback_map()[*it]->PhaseEntry(data);
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for phase exit event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_phase_exit_data_t* data, PluginKey key) {

  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->PhaseExit != 0)
      Tau_get_plugin_callback_map()[*it]->PhaseExit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_target event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_target_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptTarget != 0)
      Tau_get_plugin_callback_map()[id]->OmptTarget(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_target_data_op event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_target_data_op_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptTargetDataOp != 0)
      Tau_get_plugin_callback_map()[id]->OmptTargetDataOp(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the ompt_target_submit event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_target_submit_data_t* data, PluginKey key) {

  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptTargetSubmit != 0)
      Tau_get_plugin_callback_map()[id]->OmptTargetSubmit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks ompt_finalize event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_ompt_finalize_data_t* data, PluginKey key) {
  unsigned int ev = key.plugin_event;
  for(unsigned int i = 0; i < plugins_for_ompt_event[ev].size(); ++i) {
    unsigned int id = plugins_for_ompt_event[ev][i];
    if (Tau_get_plugin_callback_map()[id]->OmptFinalize != 0)
      Tau_get_plugin_callback_map()[id]->OmptFinalize(data);
  }
  plugins_for_ompt_event[ev].destroy();
}

/* GPU EVENTS BEGIN */
/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks gpu_init event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_gpu_init_data_t* data, PluginKey key) {
  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->GpuInit != 0)
      Tau_get_plugin_callback_map()[*it]->GpuInit(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks gpu_finalize event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_gpu_finalize_data_t* data, PluginKey key) {
  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->GpuFinalize != 0)
      Tau_get_plugin_callback_map()[*it]->GpuFinalize(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks gpu_kernel_exec event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_gpu_kernel_exec_data_t* data, PluginKey key) {
  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->GpuKernelExec != 0)
      Tau_get_plugin_callback_map()[*it]->GpuKernelExec(data);
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks gpu_memcpy event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_gpu_memcpy_data_t* data, PluginKey key) {
  for(std::set<unsigned int>::iterator it = Tau_get_plugins_for_named_specific_event()[key].begin(); it != Tau_get_plugins_for_named_specific_event()[key].end(); it++) {
    if (Tau_get_plugin_callback_map()[*it]->GpuMemcpy != 0)
      Tau_get_plugin_callback_map()[*it]->GpuMemcpy(data);
  }
}

/* GPU EVENTS END */

/* Actually do the invocation */
void Tau_util_do_invoke_callbacks(Tau_plugin_event event, PluginKey key, const void * data) {

  switch(event) {
    case TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_registration_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_METADATA_REGISTRATION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_metadata_registration_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_POST_INIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_post_init_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_MPIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_mpit_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_DUMP: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_dump_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_FUNCTION_ENTRY: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_entry_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_FUNCTION_EXIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_exit_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_current_timer_exit_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_FUNCTION_FINALIZE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_finalize_data_t*)data, key);
      break;
    }
     case TAU_PLUGIN_EVENT_SEND: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_send_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_RECV: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_recv_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_ATOMIC_EVENT_REGISTRATION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_atomic_event_registration_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_atomic_event_trigger_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_pre_end_of_execution_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_END_OF_EXECUTION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_end_of_execution_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_INTERRUPT_TRIGGER: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_interrupt_trigger_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_TRIGGER: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_trigger_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_PHASE_ENTRY: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_phase_entry_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_PHASE_EXIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_phase_exit_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_PARALLEL_BEGIN: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_parallel_begin_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_PARALLEL_END: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_parallel_end_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_TASK_CREATE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_task_create_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_TASK_SCHEDULE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_task_schedule_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_IMPLICIT_TASK: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_implicit_task_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_THREAD_BEGIN: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_thread_begin_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_THREAD_END: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_thread_end_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_WORK: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_work_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_MASTER: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_master_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_IDLE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_idle_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_SYNC_REGION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_sync_region_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_mutex_acquire_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRED: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_mutex_acquired_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_MUTEX_RELEASED: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_mutex_released_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_TARGET: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_target_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_TARGET_DATA_OP: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_target_data_op_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_TARGET_SUBMIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_target_submit_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_OMPT_FINALIZE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_ompt_finalize_data_t*)data, key);
      break;
    }
    /* GPU EVENTS START */
    case TAU_PLUGIN_EVENT_GPU_INIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_gpu_init_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_GPU_FINALIZE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_gpu_finalize_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_GPU_KERNEL_EXEC: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_gpu_kernel_exec_data_t*)data, key);
      break;
    }
    case TAU_PLUGIN_EVENT_GPU_MEMCPY: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_gpu_memcpy_data_t*)data, key);
      break;
    }
    /* GPU EVENTS END */
   default: {
      perror("Someone forgot to implement an event for plugins...\n");
      abort();
    }
  }
}

/*****************************************************************************************************************************
 * Wrapper function that calls the actual callback invocation function based on the event type
 ******************************************************************************************************************************/
extern "C" void Tau_util_invoke_callbacks_for_trigger_event(Tau_plugin_event event, size_t hash, void * data_) {

  PluginKey key_(event, hash);

  Tau_plugin_event_trigger_data_t data;
  data.data = data_;

  if(!Tau_get_plugins_for_named_specific_event()[key_].empty()) {
    Tau_util_do_invoke_callbacks(event, key_, &data);
  } else {
    PluginKey key(event, star_hash);
    Tau_util_do_invoke_callbacks(event, key, &data);
  }
}

/*****************************************************************************************************************************
 * Wrapper function that calls the actual callback invocation function based on the event type
 ******************************************************************************************************************************/
extern "C" void Tau_util_invoke_callbacks(Tau_plugin_event event, const char * specific_event_name, const void * data) {


  size_t hash_ = Tau_util_return_hash_of_string(specific_event_name);
  size_t hash;
  const char * matching_regex = Tau_check_for_matching_regex(specific_event_name);

  PluginKey key_(event, hash_);

  auto it_ = Tau_get_plugins_for_named_specific_event().find(key_);
  if(it_ != Tau_get_plugins_for_named_specific_event().end() && !it_->second.empty()) {
     hash = hash_;
  } else if (matching_regex != NULL) {
     size_t hash__ = Tau_util_return_hash_of_string(matching_regex);
     PluginKey key__(event, hash__);
     auto it__ = Tau_get_plugins_for_named_specific_event().find(key__);
     if(it__ == Tau_get_plugins_for_named_specific_event().end() || it__->second.empty()) {
       hash = star_hash;
     } else {
       hash = hash__;
     }
  } else {
     hash = star_hash;
  }

  PluginKey key(event, hash);
  Tau_util_do_invoke_callbacks(event, key, data);
}

/*****************************************************************************************************************************
 * Clean up all plugins by closing all opened dynamic libraries and free associated structures
 *******************************************************************************************************************************/
int Tau_util_cleanup_all_plugins() {

  PluginManager* plugin_manager = Tau_util_get_plugin_manager();

  Tau_plugin * temp_plugin;
  Tau_plugin_callback_t * temp_callback;

  Tau_plugin * plugin = (plugin_manager->plugin_list)->head;
  Tau_plugin_callback_t * callback = (plugin_manager->callback_list)->head;

  /*Two separate while loops to handle the weird case that a plugin is loaded but doesn't register anything*/
  while(plugin) {
    temp_plugin = plugin;

    plugin = temp_plugin->next;

    /*Close the dynamic library*/
#ifndef TAU_WINDOWS
    if(temp_plugin->handle)
      dlclose(temp_plugin->handle);
#endif /* TAU_WINDOWS */

    temp_plugin->next = NULL;

    free(temp_plugin);
  }

  while(callback) {
    temp_callback = callback;
    callback = temp_callback->next;
    temp_callback->next = NULL;

    free(temp_callback);
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////
// Plugin Send/Recv Events
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_plugin_sendmsg(long unsigned int type, long unsigned int destination, long unsigned int length, long unsigned int remoteid) {
    Tau_plugin_event_send_data plugin_data;
    plugin_data.message_tag = type;
    plugin_data.destination = destination;
    plugin_data.bytes_sent = length;
    plugin_data.tid = RtsLayer::myThread();
    double timeStamp[TAU_MAX_COUNTERS] = { 0 };
    RtsLayer::getUSecD(plugin_data.tid, timeStamp);
    plugin_data.timestamp = (unsigned long)(timeStamp[0]);
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_SEND, "*", &plugin_data);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_plugin_recvmsg(long unsigned int type, long unsigned int source, long unsigned int length, long unsigned int remoteid) {
    Tau_plugin_event_recv_data plugin_data;
    plugin_data.message_tag = type;
    plugin_data.source = source;
    plugin_data.bytes_received = length;
    plugin_data.tid = RtsLayer::myThread();
    double timeStamp[TAU_MAX_COUNTERS] = { 0 };
    RtsLayer::getUSecD(plugin_data.tid, timeStamp);
    plugin_data.timestamp = (unsigned long)(timeStamp[0]);
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_RECV, "*", &plugin_data);
}

