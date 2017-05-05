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
#include <stdarg.h>
#include <string.h>

#include <dlfcn.h>

#define TAU_NAME_LENGTH 1024
#define TAU_VERBOSE printf

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

/*Create and return a new plugin manager if plugin system is un-initialized
 * If it is already initialized, return a reference to the same plugin manager - Singleton Pattern*/
PluginManager* Tau_util_get_plugin_manager() {
  static PluginManager * plugin_manager = NULL;
  static int is_plugin_system_initialized = 0;
  
  if(!is_plugin_system_initialized) {
    plugin_manager = (PluginManager*)malloc(sizeof(PluginManager));
    plugin_manager->plugin_list = (PluginList *)malloc(sizeof(PluginList));
    (plugin_manager->plugin_list)->head = NULL;
    plugin_manager->role_hook_list = (PluginRoleHookList*)malloc(sizeof(PluginRoleHookList));
    (plugin_manager->role_hook_list)->head = NULL;
    is_plugin_system_initialized = 1;
  }

  return plugin_manager;
}

/*Initializes the plugin system by loading all and registering plugins*/
int Tau_initialize_plugin_system() {
  return(Tau_util_load_and_register_plugins(Tau_util_get_plugin_manager()));
}

/* 
 * Load a list of plugins at TAU init, given following environment variables:
 *  - TAU_PLUGINS_NAMES
 *  - TAU_PLUGINS_PATH
 */
int Tau_util_load_and_register_plugins(PluginManager* plugin_manager)
{
  char *pluginpath = NULL;
  char *listpluginsnames = NULL;
  char *fullpath = NULL;
  char *token = NULL;
  char *pluginname = NULL;
  char *initFuncName = NULL;

  pluginpath = getenv(TAU_PLUGIN_PATH);
  listpluginsnames = getenv(TAU_PLUGINS);

  if(pluginpath == NULL|| listpluginsnames == NULL) {
    printf("TAU: One or more of the environment variable(s) TAU_PLUGINS_PATH: %s, TAU_PLUGINS_NAMES: %s are empty\n", pluginpath, listpluginsnames); 
    return -1;
  }

  token = strtok(listpluginsnames,":"); 
  TAU_VERBOSE("TAU: Trying to load plugin with name %s\n", token);

  fullpath = (char*)calloc(TAU_NAME_LENGTH, sizeof(char));

  while(token != NULL)
  {
    TAU_VERBOSE("TAU: Loading plugin: %s\n", token);
    strcpy(fullpath, "");
    strcpy(fullpath,pluginpath);
    strcat(fullpath,token);
    TAU_VERBOSE("TAU: Full path for the current plugin: %s\n", fullpath);
   
    void* handle = Tau_util_load_plugin(token, fullpath, plugin_manager);

    if (handle) {
      handle = Tau_util_register_plugin(token, handle, plugin_manager);
      if(!handle) //Plugin registration failed. Bail.
        return -1;
      TAU_VERBOSE("TAU: Successfully called the init func of plugin: %s\n", token);
    } else {
      /*Plugin loading failed for some reason*/
      return -1;
    }
    token = strtok(NULL, ":");
  }

  free(fullpath);
  return 0;
}

/*Uses dlsym to find a function:TAU_PLUGIN_INIT_FUNC that the plugin MUST implement in order to register itself*/
void* Tau_util_register_plugin(const char *name, void* handle, PluginManager* plugin_manager) {
  PluginInitFunc init_func = (PluginInitFunc) dlsym(handle, TAU_PLUGIN_INIT_FUNC);

  if(!init_func) {
    printf("TAU: Failed to retrieve TAU_PLUGIN_INIT_FUNC from plugin %s with error:%s\n", name, dlerror());
    dlclose(handle); //Replace with Tau_plugin_cleanup();
    return NULL;
  }

  int return_val = init_func(plugin_manager);
  if(return_val < 0) {
    printf("TAU: Call to init func for plugin %s returned failure error code %d\n", name, return_val);
    dlclose(handle); //Replace with Tau_plugin_cleanup();
    return NULL;
  } 
  return handle;
}

/*Given a plugin name and fullpath, it loads a plugin and returns a handle to the opened DSO*/
void* Tau_util_load_plugin(const char *name, const char *path, PluginManager* plugin_manager) {
  void* handle = dlopen(path, RTLD_NOW);
  
  if (handle) {
    Plugin* plugin = (Plugin *)malloc(sizeof(Plugin));
    strcpy(plugin->plugin_name, name);
    plugin->handle = handle;
    plugin->next = (plugin_manager->plugin_list)->head;
    (plugin_manager->plugin_list)->head = plugin;

    TAU_VERBOSE("TAU: Successfully loaded plugin: %s\n", name);

    return handle;    
  } else {
    printf("TAU: Failed loading %s plugin with error: %s\n", name, dlerror());
    return NULL;
  }
}

/*Add role hook to the list of role_name:role_hook registered*/
extern "C" void Tau_util_plugin_manager_register_role_hook(PluginManager* plugin_manager, const char* role_name, PluginRoleHook role_hook) {
  PluginRoleHookNode* node = (PluginRoleHookNode*)malloc(sizeof(PluginRoleHookNode));
  strcpy(node->role_name, role_name);
  node->role_hook = role_hook;
  node->next = (plugin_manager->role_hook_list)->head;
  (plugin_manager->role_hook_list)->head = node;
}

/*Apply all role hooks for a given role_name - There may be more than one*/
extern "C" void Tau_util_apply_role_hook(PluginManager* plugin_manager, const char* role_name, int argc, void **argv) {
  int returnVal;

  PluginRoleHookNode* role_plugin = (plugin_manager->role_hook_list)->head;
  Plugin* plugin = (plugin_manager->plugin_list)->head;

  while(role_plugin) {
    if(strcmp(role_name, role_plugin->role_name) == 0) {
      returnVal = role_plugin->role_hook(argc, argv);
      if(returnVal)
        printf("TAU: Failure encountered when invoking role function of plugin: %s\n", plugin->plugin_name);
    }
    role_plugin = role_plugin->next;
    plugin = plugin->next;
  }
}

/*Initialize Tau_plugin_callbacks structure with default values. Only this function changes when more events are added*/
extern "C" void Tau_util_init_tau_plugin_callbacks(Tau_plugin_callbacks * cb) {
  cb->FunctionRegistrationComplete = 0;
  cb->FunctionInvocation = 0;
  cb->FunctionExit = 0;
  cb->AtomicEventRegistration = 0;
  cb->AtomicEventTrigger = 0;
  cb->EndOfExecution = 0;
}

/* Register callbacks associated with well defined events defined in struct Tau_plugin_callbacks*/
extern "C" void Tau_util_plugin_register_callbacks(Tau_plugin_callbacks cb) {
  PluginManager* plugin_manager = Tau_util_get_plugin_manager();
  
  
}

/*Clean up all plugins and free associated structures*/ 
int Tau_util_cleanup_all_plugins(PluginManager* plugin_manager) {
  PluginRoleHookNode* temp_role_plugin;
  Plugin* temp_plugin;

  PluginRoleHookNode* role_plugin = (plugin_manager->role_hook_list)->head;
  Plugin* plugin = (plugin_manager->plugin_list)->head;
  
  while(role_plugin) {
    temp_role_plugin = role_plugin;
    role_plugin = temp_role_plugin->next;
    temp_role_plugin->next = NULL;
    free(temp_role_plugin);
  }   
  
  while(plugin) {
    temp_plugin = plugin;
    plugin = temp_plugin->next;
    if(temp_plugin->handle)
      dlclose(temp_plugin->handle);
    temp_plugin->next = NULL;
    free(temp_plugin);
  }   
  return 0;
}


