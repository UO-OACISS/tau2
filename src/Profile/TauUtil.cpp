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
#include <stdarg.h>
#include <string.h>
#include <Profile/Profiler.h>
#include <TauMetaData.h>
#include <Profiler.h>

#ifndef TAU_WINDOWS
#include <dlfcn.h>
#else
#define strtok_r(a,b,c) strtok(a,b)
#endif /* TAU_WINDOWS */

Tau_plugin_callbacks_active_t Tau_plugins_enabled;

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
  bool excluded;			///< Is function excluded from profiling?
};

struct HashTable : public TAU_HASH_MAP<unsigned long, HashNode*>
{
  HashTable() {
    Tau_init_initializeTAU();
  }
  virtual ~HashTable() {
    Tau_destructor_trigger();
  }
};

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
      HashNode * node;
      tau_bfd_handle_t & bfdUnitHandle = TheBfdUnitHandle();
     
      node = TheHashTable()[addr];
      if (!node) {
        node = new HashNode;
        node->fi = NULL;
        node->excluded = false;
        
        TheHashTable()[addr] = node;
      }
      
      Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, node->info);

      if(node && node->info.filename && node->info.funcname && node->info.lineno) {
        sprintf(resolved_address, "%s %s [{%s} {%d, 0}]", region_type, node->info.funcname, node->info.filename, node->info.lineno);
      } else if(node && node->info.filename && node->info.funcname) {
        sprintf(resolved_address, "%s %s [{%s}]", region_type, node->info.funcname, node->info.filename);
      } else if(node && node->info.funcname) {
        sprintf(resolved_address, "%s %s", region_type, node->info.funcname);
      } else {
        sprintf(resolved_address, "OpenMP %s __UNKNOWN__", region_type);
      }
      #else 
        sprintf(resolved_address, "OpenMP %s __UNKNOWN__", region_type);
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
     
      RtsLayer::LockDB();  
      node = TheHashTable()[addr];
      if (!node) {
        node = new HashNode;
        node->fi = NULL;
        node->excluded = false;
        
        TheHashTable()[addr] = node;

        Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, node->info);
      }
      RtsLayer::UnLockDB(); 

      if(node && node->info.filename && node->info.funcname && node->info.lineno) {
        sprintf(resolved_address, "%s [{%s} {%d, 0}]", node->info.funcname, node->info.filename, node->info.lineno);
      } else if(node && node->info.filename && node->info.funcname) {
        sprintf(resolved_address, "%s [{%s}]", node->info.funcname, node->info.filename);
      } else if(node && node->info.funcname) {
        sprintf(resolved_address, "%s", node->info.funcname);
      } else {
        sprintf(resolved_address, "__UNKNOWN__");
      }
      #else
        sprintf(resolved_address, "__UNKNOWN__");
      #endif /*TAU_BFD*/
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
  *plugin_name = (char*)malloc(1024*sizeof(char));
  pos_left = strchr(token, '(');
  pos_right = strchr(token, ')');

  if(pos_left == NULL && pos_right == NULL) {
    strcpy(*plugin_name, token);
    return 0;
  } else if (pos_left == NULL || pos_right == NULL) {
    return -1; //Bad plugin name
  }

  *plugin_args = (char**)malloc(10*sizeof(char*)); //Maximum of 10 args supported for now
  arg_string = (char*)malloc(1024*sizeof(char));

  length_of_arg_string = (pos_right - pos_left) - 1;

  strncpy(arg_string, pos_left+1, length_of_arg_string);
  strncpy(*plugin_name, token, (pos_left-token));

  arg_token = strtok_r(arg_string, ",", &save_ptr);

  int i = 0;
  /*Grab and pack, and count all the arguments to the plugin*/
  while(arg_token != NULL) {
    (*plugin_num_args)++;
    (*plugin_args)[i] = (char*)malloc(1024*sizeof(char));
    strcpy((*plugin_args)[i], arg_token);
    arg_token = strtok_r(NULL, ",", &save_ptr);
    i++;
  }

  TAU_VERBOSE("TAU PLUGIN: Arg string and count for token %s are %s and %d\n", token, arg_string, *plugin_num_args);

  return 0;
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
  char *fullpath = NULL;
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
  
  strcpy(pluginpath, TauEnv_get_plugins_path());
  strcpy(listpluginsnames, TauEnv_get_plugins());

  /*Individual plugin names are separated by a ":"*/
  token = strtok_r(listpluginsnames,":", &save_ptr); 
  TAU_VERBOSE("TAU: Trying to load plugin with name %s\n", token);

  fullpath = (char*)calloc(TAU_NAME_LENGTH, sizeof(char));

  while(token != NULL)
  {
    TAU_VERBOSE("TAU: Loading plugin: %s\n", token);
    strcpy(fullpath, "");
    strcpy(fullpath,pluginpath);
    if (Tau_util_parse_plugin_token(token, &plugin_name, &plugin_args, &plugin_num_args)) {
      printf("TAU: Plugin name specification does not match form <plugin_name1>(<plugin_arg1>,<plugin_arg2>):<plugin_name2>(<plugin_arg1>,<plugin_arg2>) for: %s\n",token);
      return -1;
    }

#ifndef TAU_WINDOWS
    sprintf(fullpath, "%s/%s", pluginpath, plugin_name);
#else
    sprintf(fullpath, "%s\\%s", pluginpath, plugin_name);
#endif

    TAU_VERBOSE("TAU: Full path for the current plugin: %s\n", fullpath);
   
    /*Return a handle to the loaded dynamic object*/
    void* handle = Tau_util_load_plugin(plugin_name, fullpath, plugin_manager);

    if (handle) {
      /*If handle is NOT NULL, register the plugin's handlers for various supported events*/
      handle = Tau_util_register_plugin(plugin_name, plugin_args, plugin_num_args, handle, plugin_manager);
     
      /*Plugin registration failed. Bail*/
      if(!handle) return -1;
      TAU_VERBOSE("TAU: Successfully called the init func of plugin: %s\n", token);

    } else {
      /*Plugin loading failed for some reason*/
      return -1;
    }

    token = strtok_r(NULL, ":", &save_ptr);
  }
  Tau_metadata_push_to_plugins();

  free(fullpath);
  return 0;
}

/**************************************************************************************************************************
 * Use dlsym to find a function : TAU_PLUGIN_INIT_FUNC that the plugin MUST implement in order to register itself.
 * If plugin registration succeeds, then the callbacks for that plugin have been added to the plugin manager's callback list
 * ************************************************************************************************************************/
void* Tau_util_register_plugin(const char *name, char **args, int num_args, void* handle, PluginManager* plugin_manager) {
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

  int return_val = init_func(num_args, args);
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
  void* handle = dlopen(path, RTLD_NOW);
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
  cb->FunctionFinalize = 0;
  cb->PhaseEntry = 0;
  cb->PhaseExit = 0;
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
  dest->FunctionFinalize = src->FunctionFinalize;
  dest->PhaseEntry = src->PhaseEntry;
  dest->PhaseExit = src->PhaseExit;
}


/**************************************************************************************************************************
 * Register callbacks associated with well defined events defined in struct Tau_plugin_callbacks
 **************************************************************************************************************************/
extern "C" void Tau_util_plugin_register_callbacks(Tau_plugin_callbacks * cb) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();

  Tau_plugin_callback_t * callback = (Tau_plugin_callback_t *)malloc(sizeof(Tau_plugin_callback_t));
  Tau_util_make_callback_copy(&(callback->cb), cb);
  callback->next = (plugin_manager->callback_list)->head;
  (plugin_manager->callback_list)->head = callback;

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
  if (cb->PhaseEntry != 0) { Tau_plugins_enabled.phase_entry = 1; }
  if (cb->PhaseExit != 0) { Tau_plugins_enabled.phase_exit = 1; }
  
}


/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the function registration event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_registration_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.FunctionRegistrationComplete != 0) {
     callback->cb.FunctionRegistrationComplete(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the mpit event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_mpit_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.Mpit != 0) {
     callback->cb.Mpit(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the dump event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_dump_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.Dump != 0) {
     callback->cb.Dump(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the function entry event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_entry_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.FunctionEntry != 0) {
     callback->cb.FunctionEntry(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the function exit event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_exit_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.FunctionExit != 0) {
     callback->cb.FunctionExit(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the "current timer" exit event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_current_timer_exit_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.CurrentTimerExit != 0) {
     callback->cb.CurrentTimerExit(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the send event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_send_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.Send != 0) {
     callback->cb.Send(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the recv event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_recv_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.Recv != 0) {
     callback->cb.Recv(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the metadata registration event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_metadata_registration_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.MetadataRegistrationComplete != 0) {
     callback->cb.MetadataRegistrationComplete(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the post init event
 ***************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_post_init_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.PostInit != 0) {
     callback->cb.PostInit(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the atomic event registration event
 ****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_atomic_event_registration_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.AtomicEventRegistrationComplete != 0) {
     callback->cb.AtomicEventRegistrationComplete(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the atomic event trigger event
 *****************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_atomic_event_trigger_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.AtomicEventTrigger != 0) {
     callback->cb.AtomicEventTrigger(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the pre end of execution event
 ******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_pre_end_of_execution_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.PreEndOfExecution != 0) {
     callback->cb.PreEndOfExecution(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 * Overloaded function that invokes all registered callbacks for the end of execution event
 ******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_end_of_execution_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.EndOfExecution != 0) {
     callback->cb.EndOfExecution(data);
   }
   callback = callback->next;
  }
}

/*****************************************************************************
 * Overloaded function that invokes all registered callbacks for the 
 * finalize event
 *****************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_function_finalize_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.FunctionFinalize != 0) {
     callback->cb.FunctionFinalize(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for interrupt trigger event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_interrupt_trigger_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.InterruptTrigger != 0) {
     callback->cb.InterruptTrigger(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for phase entry event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_phase_entry_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.PhaseEntry != 0) {
     callback->cb.PhaseEntry(data);
   }
   callback = callback->next;
  }
}

/**************************************************************************************************************************
 *  Overloaded function that invokes all registered callbacks for phase exit event
 *******************************************************************************************************************************/
void Tau_util_invoke_callbacks_(Tau_plugin_event_phase_exit_data_t* data) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  Tau_plugin_callback_list * callback_list = plugin_manager->callback_list;
  Tau_plugin_callback_t * callback = callback_list->head;

  while(callback != NULL) {
   if(callback->cb.PhaseExit != 0) {
     callback->cb.PhaseExit(data);
   }
   callback = callback->next;
  }
}

/*****************************************************************************************************************************
 * Wrapper function that calls the actual callback invocation function based on the event type
 ******************************************************************************************************************************/
extern "C" void Tau_util_invoke_callbacks(Tau_plugin_event event, const void * data) {

  switch(event) {
    case TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_registration_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_METADATA_REGISTRATION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_metadata_registration_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_POST_INIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_post_init_data_t*)data);
      break;
    }  
    case TAU_PLUGIN_EVENT_MPIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_mpit_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_DUMP: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_dump_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_FUNCTION_ENTRY: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_entry_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_FUNCTION_EXIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_exit_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_current_timer_exit_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_FUNCTION_FINALIZE: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_function_finalize_data_t*)data);
      break;
    }
     case TAU_PLUGIN_EVENT_SEND: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_send_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_RECV: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_recv_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_ATOMIC_EVENT_REGISTRATION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_atomic_event_registration_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_atomic_event_trigger_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_pre_end_of_execution_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_END_OF_EXECUTION: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_end_of_execution_data_t*)data);
      break;
    } 
    case TAU_PLUGIN_EVENT_INTERRUPT_TRIGGER: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_interrupt_trigger_data_t*)data);
      break;
    }
    case TAU_PLUGIN_EVENT_PHASE_ENTRY: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_phase_entry_data_t*)data);
      break;
    }
    case TAU_PLUGIN_EVENT_PHASE_EXIT: {
      Tau_util_invoke_callbacks_((Tau_plugin_event_phase_exit_data_t*)data);
      break;
    }
   default: {
      perror("Someone forgot to implement an event for plugins...\n");
      abort();
    }
  }
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
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_SEND, &plugin_data);
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
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_RECV, &plugin_data);
}

