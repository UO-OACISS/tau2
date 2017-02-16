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
#include <stdarg.h>
#include <string.h>

#include <dlfcn.h>


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

/*
 * Load a plugin with its given name and path
 */
int Tau_util_load_plugin(char *name, char *path, int num_args, void **args)
{
  char *fullname;
  char *fullpath;
  char *initFuncName;
  
  strcat(path, name);

  sprintf(fullpath, "%s.so", path);

  if(pds == NULL)
    pds  = (PluginDiscoveryState *)malloc(sizeof(PluginDiscoveryState)); 

  void *handle = dlopen(fullpath, RTLD_NOW);
  //dstring_free(slashedpath);
  
  if (handle) {
    //PluginHandleList* handle_node = mem_alloc(sizeof(*handle_node));
    PluginHandleList* handle_node = (PluginHandleList *)malloc(sizeof(PluginHandleList));
    handle_node->handle = handle;
    handle_node->next = pds->handle_list;
    pds->handle_list = handle_node;
  } else {
    printf("Error loading DSO: %s\n", dlerror());
    return -1;
  } 

  sprintf(initFuncName, "plugin_%s", name);  

  /* Get symbol of plugin entry point */
  void (*fn)(int num_args, void **args) = (void (*)(int num_args, void **))dlsym(handle, initFuncName);

  if(!fn) {
    fprintf(stdout, "Error loading plugin function: %s\n", dlerror());
    dlclose(handle);
    return -1;
  }

  /* Call plugin function  */
  fn(num_args, args);

  return 1;
}

int Tau_util_close_plugin()
{

  return 1;
}

/*
 * Clean up all plugins and free associated structures
 */
void Tau_util_cleanup_plugins(void *vpds)
{

  PluginDiscoveryState* pds = (PluginDiscoveryState*)vpds;
  PluginHandleList* node = pds->handle_list;

  while (node) {
    PluginHandleList* next = node->next;
    dlclose(node->handle);
    free(node);
    node = next;
  }
  
  free(pds);
}


