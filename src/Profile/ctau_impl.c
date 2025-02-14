#include "Python.h"
#include "compile.h"
#include "frameobject.h"
#include "structseq.h"
#include <TAU.h>

// Python 3 compatibility
#if PY_MAJOR_VERSION >= 3
#define PyString_FromString(s) PyUnicode_FromString(s)
#define PyString_AsString(s) PyUnicode_AsUTF8(s)
#define PyInt_FromLong(l) PyLong_FromLong(l)
#define staticforward static
#define statichere static
#define PyString_Check(s) PyUnicode_Check(s)
#define PyString_AS_STRING(s) PyUnicode_AsUTF8(s)
#define PyString_FromFormat PyUnicode_FromFormat
#endif
#ifndef Py_TYPE
#define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif
#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(type, size) PyObject_HEAD_INIT(type) size,
#endif

// Python 3.11 compatibility
// This is from https://docs.python.org/3/whatsnew/3.11.html
// to provide backwards compatibility with Python <3.8
// (Python 3.8 and later provides PyFrame_GetCode natively)
// Python 3.11 and later remove the f_code field of frame
// requiring accessor method instead
#if PY_VERSION_HEX < 0x030900B1
static inline PyCodeObject* PyFrame_GetCode(PyFrameObject *frame)
{
    Py_INCREF(frame->f_code);
    return frame->f_code;
}
#endif

/************************ rotatingtree.h *************************/

/* "Rotating trees" (Armin Rigo)
 *
 * Google "splay trees" for the general idea.
 *
 * It's a dict-like data structure that works best when accesses are not
 * random, but follow a strong pattern.  The one implemented here is for
 * access patterns where the same small set of keys is looked up over
 * and over again, and this set of keys evolves slowly over time.
 */

#include <stdlib.h>

#define EMPTY_ROTATING_TREE       ((rotating_node_t *)NULL)

typedef struct rotating_node_s rotating_node_t;
typedef int (*rotating_tree_enum_fn) (rotating_node_t *node, void *arg);

struct rotating_node_s {
  void *key;
  rotating_node_t *left;
  rotating_node_t *right;
};

void RotatingTree_Add(rotating_node_t **root, rotating_node_t *node);
rotating_node_t* RotatingTree_Get(rotating_node_t **root, void *key);
int RotatingTree_Enum(rotating_node_t *root, rotating_tree_enum_fn enumfn,
		      void *arg);


/************************ rotatingtree.c *************************/

#define KEY_LOWER_THAN(key1, key2)  ((char*)(key1) < (char*)(key2))

/* The randombits() function below is a fast-and-dirty generator that
 * is probably irregular enough for our purposes.  Note that it's biased:
 * I think that ones are slightly more probable than zeroes.  It's not
 * important here, though.
 */

static unsigned int random_value = 1;
static unsigned int random_stream = 0;

static int randombits(int bits) {
  int result;
  if (random_stream < (1U << bits)) {
    random_value *= 1082527;
    random_stream = random_value;
  }
  result = random_stream & ((1<<bits)-1);
  random_stream >>= bits;
  return result;
}


/* Insert a new node into the tree.
   (*root) is modified to point to the new root. */
void RotatingTree_Add(rotating_node_t **root, rotating_node_t *node) {
  while (*root != NULL) {
    if (KEY_LOWER_THAN(node->key, (*root)->key))
      root = &((*root)->left);
    else
      root = &((*root)->right);
  }
  node->left = NULL;
  node->right = NULL;
  *root = node;
}

/* Locate the node with the given key.  This is the most complicated
   function because it occasionally rebalances the tree to move the
   resulting node closer to the root. */
rotating_node_t *RotatingTree_Get(rotating_node_t **root, void *key) {
  if (randombits(3) != 4) {
    /* Fast path, no rebalancing */
    rotating_node_t *node = *root;
    while (node != NULL) {
      if (node->key == key)
	return node;
      if (KEY_LOWER_THAN(key, node->key))
	node = node->left;
      else
	node = node->right;
    }
    return NULL;
  } else {
    rotating_node_t **pnode = root;
    rotating_node_t *node = *pnode;
    rotating_node_t *next;
    int rotate;
    if (node == NULL)
      return NULL;
    while (1) {
      if (node->key == key)
	return node;
      rotate = !randombits(1);
      if (KEY_LOWER_THAN(key, node->key)) {
	next = node->left;
	if (next == NULL)
	  return NULL;
	if (rotate) {
	  node->left = next->right;
	  next->right = node;
	  *pnode = next;
	}
	else
	  pnode = &(node->left);
      }
      else {
	next = node->right;
	if (next == NULL)
	  return NULL;
	if (rotate) {
	  node->right = next->left;
	  next->left = node;
	  *pnode = next;
	}
	else
	  pnode = &(node->right);
      }
      node = next;
    }
  }
}

/* Enumerate all nodes in the tree.  The callback enumfn() should return
   zero to continue the enumeration, or non-zero to interrupt it.
   A non-zero value is directly returned by RotatingTree_Enum(). */
int RotatingTree_Enum(rotating_node_t *root, rotating_tree_enum_fn enumfn, void *arg) {
  int result;
  rotating_node_t *node;
  while (root != NULL) {
    result = RotatingTree_Enum(root->left, enumfn, arg);
    if (result != 0) return result;
    node = root->right;
    result = enumfn(root, arg);
    if (result != 0) return result;
    root = node;
  }
  return 0;
}


/************************ ************** *************************/



#if !defined(HAVE_LONG_LONG)
#error "This module requires long longs!"
#endif

/*** Selection of a high-precision timer ***/

#ifdef MS_WINDOWS

#include <windows.h>

static PY_LONG_LONG
hpTimer(void)
{
  LARGE_INTEGER li;
  QueryPerformanceCounter(&li);
  return li.QuadPart;
}

static double
hpTimerUnit(void)
{
  LARGE_INTEGER li;
  if (QueryPerformanceFrequency(&li))
    return 1.0 / li.QuadPart;
  else
    return 0.000001;  /* unlikely */
}

#else  /* !MS_WINDOWS */

// In Python 3.9 and later, HAVE_GETTIMEOFDAY is no longer defined
// (but structure of time variables is as if it were)
// so we define it ourselves.
#if (PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >= 9)
#define HAVE_GETTIMEOFDAY
#endif

#ifndef HAVE_GETTIMEOFDAY
#error "This module requires gettimeofday() on non-Windows platforms!"
#endif

#if (defined(PYOS_OS2) && defined(PYCC_GCC))
#include <sys/time.h>
#else
#include <sys/resource.h>
#include <sys/times.h>
#if PY_VERSION_HEX > 0x030D0000 // Python versions after 3.13
    // Include sys/time.h because pyport.h no longer does so.
    #include <sys/time.h>
#endif
#endif
static PY_LONG_LONG
hpTimer(void)
{
  struct timeval tv;
  PY_LONG_LONG ret;
#ifdef GETTIMEOFDAY_NO_TZ
  gettimeofday(&tv);
#else
  gettimeofday(&tv, (struct timezone *)NULL);
#endif
  ret = tv.tv_sec;
  ret = ret * 1000000 + tv.tv_usec;
  return ret;
}

static double
hpTimerUnit(void)
{
  return 0.000001;
}

#endif  /* MS_WINDOWS */

/************************************************************/
/* Written by Brett Rosen and Ted Czotter */

struct _ProfilerEntry;

/* represents a function called from another function */
typedef struct _ProfilerSubEntry {
  rotating_node_t header;
  PY_LONG_LONG tt;
  PY_LONG_LONG it;
  long callcount;
  long recursivecallcount;
  long recursionLevel;
} ProfilerSubEntry;

/* represents a function or user defined block */
typedef struct _ProfilerEntry {
  rotating_node_t header;
  PyObject *userObj; /* PyCodeObject, or a descriptive str for builtins */
  PY_LONG_LONG tt; /* total time in this entry */
  PY_LONG_LONG it; /* inline time in this entry (not in subcalls) */
  long callcount; /* how many times this was called */
  long recursivecallcount; /* how many times called recursively */
  long recursionLevel;
  rotating_node_t *calls;
  void *fi;
} ProfilerEntry;

typedef struct _ProfilerContext {
  PY_LONG_LONG t0;
  PY_LONG_LONG subt;
  struct _ProfilerContext *previous;
  ProfilerEntry *ctxEntry;
} ProfilerContext;

typedef struct {
  PyObject_HEAD
  rotating_node_t *profilerEntries;
  ProfilerContext *currentProfilerContext;
  ProfilerContext *freelistProfilerContext;
  int flags;
  PyObject *externalTimer;
  double externalTimerUnit;
} ProfilerObject;

#define POF_ENABLED     0x001
#define POF_SUBCALLS    0x002
#define POF_BUILTINS    0x004
#define POF_NOMEMORY    0x100

staticforward PyTypeObject PyProfiler_Type;

#define PyProfiler_Check(op) PyObject_TypeCheck(op, &PyProfiler_Type)
#define PyProfiler_CheckExact(op) ((op)->ob_type == &PyProfiler_Type)

/*** External Timers ***/

#define DOUBLE_TIMER_PRECISION   4294967296.0
static PyObject *empty_tuple;

static PY_LONG_LONG CallExternalTimer(ProfilerObject *pObj)
{
  PY_LONG_LONG result;
  PyObject *o = PyObject_Call(pObj->externalTimer, empty_tuple, NULL);
  if (o == NULL) {
    PyErr_WriteUnraisable(pObj->externalTimer);
    return 0;
  }
  if (pObj->externalTimerUnit > 0.0) {
    /* interpret the result as an integer that will be scaled
       in profiler_getstats() */
    result = PyLong_AsLongLong(o);
  }
  else {
    /* interpret the result as a double measured in seconds.
       As the profiler works with PY_LONG_LONG internally
       we convert it to a large integer */
    double val = PyFloat_AsDouble(o);
    /* error handling delayed to the code below */
    result = (PY_LONG_LONG) (val * DOUBLE_TIMER_PRECISION);
  }
  Py_DECREF(o);
  if (PyErr_Occurred()) {
    PyErr_WriteUnraisable((PyObject *) pObj);
    return 0;
  }
  return result;
}

#define CALL_TIMER(pObj)	((pObj)->externalTimer ?	\
				 CallExternalTimer(pObj) :	\
				 hpTimer())

/*** ProfilerObject ***/

static PyObject *
normalizeUserObj(PyObject *obj)
{
  PyCFunctionObject *fn;
  if (!PyCFunction_Check(obj)) {
    Py_INCREF(obj);
    return obj;
  }
  /* Replace built-in function objects with a descriptive string
     because of built-in methods -- keeping a reference to
     __self__ is probably not a good idea. */
  fn = (PyCFunctionObject *)obj;

  if (fn->m_self == NULL) {
    /* built-in function: look up the module name */
    PyObject *mod = fn->m_module;
    const char * modname;
    if (mod && PyString_Check(mod)) {
      modname = PyString_AS_STRING(mod);
    }
    else if (mod && PyModule_Check(mod)) {
      modname = PyModule_GetName(mod);
      if (modname == NULL) {
	PyErr_Clear();
	modname = "__builtin__";
      }
    }
    else {
      modname = "__builtin__";
    }
    if (strcmp(modname, "__builtin__") != 0)
      return PyString_FromFormat("<%s.%s>",
				 modname,
				 fn->m_ml->ml_name);
    else
      return PyString_FromFormat("<%s>",
				 fn->m_ml->ml_name);
  }
  else {
    /* built-in method: try to return
       repr(getattr(type(__self__), __name__))
    */
    PyObject *self = fn->m_self;
    PyObject *name = PyString_FromString(fn->m_ml->ml_name);
    if (name != NULL) {
      PyObject *mo = _PyType_Lookup(self->ob_type, name);
      Py_XINCREF(mo);
      Py_DECREF(name);
      if (mo != NULL) {
	PyObject *res = PyObject_Repr(mo);
	Py_DECREF(mo);
	if (res != NULL)
	  return res;
      }
    }
    PyErr_Clear();
    return PyString_FromFormat("<built-in method %s>",
			       fn->m_ml->ml_name);
  }
}

static ProfilerEntry *newProfilerEntry(ProfilerObject *pObj, void *key, PyObject *userObj,
				       PyFrameObject *frame, char *cname) {
  char routine[4096];
  const char * co_name, * co_filename;
  int co_firstlineno;

  ProfilerEntry *self;
  void *handle = NULL;
  self = (ProfilerEntry*) malloc(sizeof(ProfilerEntry));
  if (self == NULL) {
    pObj->flags |= POF_NOMEMORY;
    return NULL;
  }
  userObj = normalizeUserObj(userObj);
  if (userObj == NULL) {
    PyErr_Clear();
    free(self);
    pObj->flags |= POF_NOMEMORY;
    return NULL;
  }
  self->header.key = key;
  self->userObj = userObj;
  self->tt = 0;
  self->it = 0;
  self->callcount = 0;
  self->recursivecallcount = 0;
  self->recursionLevel = 0;
  self->calls = EMPTY_ROTATING_TREE;

  if (frame != NULL) {
      PyCodeObject * codeObj = PyFrame_GetCode(frame);
      co_name = PyString_AsString(codeObj->co_name);
      if(codeObj->co_filename != NULL) {
        co_filename = PyString_AsString(codeObj->co_filename);
        if(co_filename == NULL) {
            co_filename = "";
        }
      } else {
        co_filename = "";
      }
      while (strchr(co_filename,'/')) {
	    co_filename = strchr(co_filename,'/')+1;
      }
      co_firstlineno = codeObj->co_firstlineno;
      snprintf (routine, sizeof(routine), "%s [{%s}{%d}]", co_name, co_filename, co_firstlineno);
      if (strcmp(co_filename,"<string>") != 0) { // suppress "? <string>"
	    TAU_PROFILER_CREATE(handle, routine, "", TAU_PYTHON);
      }
      Py_DECREF(codeObj);
  } else {
    if (strcmp (cname, "profileTimer") && strcmp (cname, "start") && strcmp (cname, "stop") && strcmp (cname, "disable")) {
      snprintf (routine, sizeof(routine), "%s", cname);
      TAU_PROFILER_CREATE(handle, routine, "", TAU_PYTHON);
    }
  }

  self->fi = handle;

  RotatingTree_Add(&pObj->profilerEntries, &self->header);

  return self;
}

static ProfilerEntry*getEntry(ProfilerObject *pObj, void *key) {
  return (ProfilerEntry*) RotatingTree_Get(&pObj->profilerEntries, key);
}

static ProfilerSubEntry *getSubEntry(ProfilerObject *pObj, ProfilerEntry *caller, ProfilerEntry* entry) {
  return (ProfilerSubEntry*) RotatingTree_Get(&caller->calls, (void *)entry);
}

static ProfilerSubEntry *newSubEntry(ProfilerObject *pObj,  ProfilerEntry *caller, ProfilerEntry* entry) {
  ProfilerSubEntry *self;
  self = (ProfilerSubEntry*) malloc(sizeof(ProfilerSubEntry));
  if (self == NULL) {
    pObj->flags |= POF_NOMEMORY;
    return NULL;
  }
  self->header.key = (void *)entry;
  self->tt = 0;
  self->it = 0;
  self->callcount = 0;
  self->recursivecallcount = 0;
  self->recursionLevel = 0;
  RotatingTree_Add(&caller->calls, &self->header);
  return self;
}

static int freeSubEntry(rotating_node_t *header, void *arg) {
  ProfilerSubEntry *subentry = (ProfilerSubEntry*) header;
  free(subentry);
  return 0;
}

static int freeEntry(rotating_node_t *header, void *arg) {
  ProfilerEntry *entry = (ProfilerEntry*) header;
  RotatingTree_Enum(entry->calls, freeSubEntry, NULL);
  Py_DECREF(entry->userObj);
  free(entry);
  return 0;
}

static void clearEntries(ProfilerObject *pObj) {
  RotatingTree_Enum(pObj->profilerEntries, freeEntry, NULL);
  pObj->profilerEntries = EMPTY_ROTATING_TREE;
  /* release the memory hold by the free list of ProfilerContexts */
  while (pObj->freelistProfilerContext) {
    ProfilerContext *c = pObj->freelistProfilerContext;
    pObj->freelistProfilerContext = c->previous;
    free(c);
  }
}


static void Stop(ProfilerObject *pObj, ProfilerContext *self, ProfilerEntry *entry) {
}



static void ptrace_enter_call(PyObject *self, void *key, PyObject *userObj, PyFrameObject *frame, char *cname) {
  /* entering a call to the function identified by 'key'
     (which can be a PyCodeObject or a PyMethodDef pointer) */
  ProfilerObject *pObj = (ProfilerObject*)self;
  ProfilerEntry *profEntry;
  ProfilerContext *pContext;


  profEntry = getEntry(pObj, key);
  if (profEntry == NULL) {
    profEntry = newProfilerEntry(pObj, key, userObj, frame, cname);
    if (profEntry == NULL) {
      return;
    }
  }

  if (profEntry->fi) {
    Tau_start_timer(profEntry->fi,0,Tau_get_thread());
  }

}

static void ptrace_leave_call(PyObject *self, void *key) {
  char *name;

  /* leaving a call to the function identified by 'key' */
  ProfilerObject *pObj = (ProfilerObject*)self;
  ProfilerEntry *profEntry;
  ProfilerContext *pContext;

  profEntry = getEntry(pObj, key);
  if (profEntry) {
    if (profEntry->fi) {
      Tau_stop_timer(profEntry->fi, Tau_get_thread());
    }
  }
  else {
    printf ("Error in TAU Python profiler: tried to stop timer but timer not found in ptrace_leave_call!\n");
  }
}



static int profiler_callback(PyObject *self, PyFrameObject *frame, int what, PyObject *arg) {
  PyCFunctionObject *fn;

  static int init = 0;
  if (init == 0) {
    if (Tau_get_node() == -1) {
        TAU_PROFILE_SET_NODE(0);
    }
    init = 1;
  }

  switch (what) {

    /* the 'frame' of a called function is about to start its execution */
  case PyTrace_CALL:
/*      printf ("Py Enter: %s\n", routine);  */
     {
         PyCodeObject * codeObj = PyFrame_GetCode(frame);
         ptrace_enter_call(self, (void *)codeObj,
		           (PyObject *)codeObj, frame, NULL);
         Py_DECREF(codeObj);
     }
    break;

    /* the 'frame' of a called function is about to finish
       (either normally or with an exception) */
  case PyTrace_RETURN:
/*      printf ("Py Exit: %s\n", routine);  */
    {
        PyCodeObject * codeObj = PyFrame_GetCode(frame);
        ptrace_leave_call(self, (void *)codeObj);
        Py_DECREF(codeObj);
    }
    break;

    /* case PyTrace_EXCEPTION:
       If the exception results in the function exiting, a
       PyTrace_RETURN event will be generated, so we don't need to
       handle it. */

#ifdef PyTrace_C_CALL	/* not defined in Python <= 2.3 */
    /* the Python function 'frame' is issuing a call to the built-in
       function 'arg' */
  case PyTrace_C_CALL:
    if ((((ProfilerObject *)self)->flags & POF_BUILTINS)
	&& PyCFunction_Check(arg)) {
      fn = (PyCFunctionObject *)arg;
/*       printf ("C Call enter! %s\n", fn->m_ml->ml_name); */
      ptrace_enter_call(self,
			((PyCFunctionObject *)arg)->m_ml,
			arg, NULL, (char *)fn->m_ml->ml_name);

    }
    break;

    /* the call to the built-in function 'arg' is returning into its
       caller 'frame' */
  case PyTrace_C_RETURN:		/* ...normally */
  case PyTrace_C_EXCEPTION:	/* ...with an exception set */
    if ((((ProfilerObject *)self)->flags & POF_BUILTINS)
	&& PyCFunction_Check(arg)) {
      fn = (PyCFunctionObject *)arg;
/*       printf ("C Call exit! %s\n", fn->m_ml->ml_name); */
      ptrace_leave_call(self,
			((PyCFunctionObject *)arg)->m_ml);

    }
    break;
#endif

  default:
    break;
  }
  return 0;
}

static int pending_exception(ProfilerObject *pObj) {
  if (pObj->flags & POF_NOMEMORY) {
    pObj->flags -= POF_NOMEMORY;
    PyErr_SetString(PyExc_MemoryError,
		    "memory was exhausted while profiling");
    return -1;
  }
  return 0;
}

/************************************************************/

static PyStructSequence_Field profiler_entry_fields[] = {
  {"code",         "code object or built-in function name"},
  {"callcount",    "how many times this was called"},
  {"reccallcount", "how many times called recursively"},
  {"totaltime",    "total time in this entry"},
  {"inlinetime",   "inline time in this entry (not in subcalls)"},
  {"calls",        "details of the calls"},
  {0}
};

static PyStructSequence_Field profiler_subentry_fields[] = {
  {"code",         "called code object or built-in function name"},
  {"callcount",    "how many times this is called"},
  {"reccallcount", "how many times this is called recursively"},
  {"totaltime",    "total time spent in this call"},
  {"inlinetime",   "inline time (not in further subcalls)"},
  {0}
};

static PyStructSequence_Desc profiler_entry_desc = {
  "ctau_impl.profiler_entry", /* name */
  NULL, /* doc */
  profiler_entry_fields,
  6
};

static PyStructSequence_Desc profiler_subentry_desc = {
  "ctau_impl.profiler_subentry", /* name */
  NULL, /* doc */
  profiler_subentry_fields,
  5
};

static int initialized;
static PyTypeObject StatsEntryType;
static PyTypeObject StatsSubEntryType;


typedef struct {
  PyObject *list;
  PyObject *sublist;
  double factor;
} statscollector_t;

static int statsForSubEntry(rotating_node_t *node, void *arg) {
  return 0;
}

static int statsForEntry(rotating_node_t *node, void *arg) {
  return 0;
}

PyDoc_STRVAR(getstats_doc, "\
getstats() -> list of profiler_entry objects\n\
\n\
Return all information collected by the profiler.\n\
Each profiler_entry is a tuple-like object with the\n\
following attributes:\n\
\n\
    code          code object\n\
    callcount     how many times this was called\n\
    reccallcount  how many times called recursively\n\
    totaltime     total time in this entry\n\
    inlinetime    inline time in this entry (not in subcalls)\n\
    calls         details of the calls\n\
\n\
The calls attribute is either None or a list of\n\
profiler_subentry objects:\n\
\n\
    code          called code object\n\
    callcount     how many times this is called\n\
    reccallcount  how many times this is called recursively\n\
    totaltime     total time spent in this call\n\
    inlinetime    inline time (not in further subcalls)\n\
");

static PyObject*profiler_getstats(ProfilerObject *pObj, PyObject* noarg) {
  return NULL;
}

static int setSubcalls(ProfilerObject *pObj, int nvalue){
  return 0;
}

static int setBuiltins(ProfilerObject *pObj, int nvalue)
{
  if (nvalue == 0)
    pObj->flags &= ~POF_BUILTINS;
  else if (nvalue > 0) {
#ifndef PyTrace_C_CALL
    PyErr_SetString(PyExc_ValueError,
		    "builtins=True requires Python >= 2.4");
    return -1;
#else
    pObj->flags |=  POF_BUILTINS;
#endif
  }
  return 0;
}

PyDoc_STRVAR(enable_doc, "\
enable(subcalls=True, builtins=True)\n\
\n\
Start collecting profiling information.\n\
If 'subcalls' is True, also records for each function\n\
statistics separated according to its current caller.\n\
If 'builtins' is True, records the time spent in\n\
built-in functions separately from their caller.\n\
");

static PyObject *profiler_enable(ProfilerObject *self, PyObject *args, PyObject *kwds) {
  int subcalls = -1;
  int builtins = -1;
  static char *kwlist[] = {"subcalls", "builtins", 0};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii:enable",
				   kwlist, &subcalls, &builtins))
    return NULL;
  if (setSubcalls(self, subcalls) < 0 || setBuiltins(self, builtins) < 0)
    return NULL;
  PyEval_SetProfile(profiler_callback, (PyObject*)self);
  self->flags |= POF_ENABLED;
  Py_INCREF(Py_None);
  return Py_None;
}

static void flush_unmatched(ProfilerObject *pObj) {
}

PyDoc_STRVAR(disable_doc, "\
disable()\n\
\n\
Stop collecting profiling information.\n\
");

static PyObject* profiler_disable(ProfilerObject *self, PyObject* noarg) {
  self->flags &= ~POF_ENABLED;
  PyEval_SetProfile(NULL, NULL);
  flush_unmatched(self);
  if (pending_exception(self))
    return NULL;
  Py_INCREF(Py_None);
  return Py_None;
}

extern void Tau_profile_exit_all_threads(void);

PyDoc_STRVAR(exitAllThreads_doc, "\
exitAllThreads()\n\
\n\
Write all information collected so far to file and disable instrumentation.\n\
");

static PyObject *profiler_exitAllThreads(ProfilerObject *pObj, PyObject* noarg) {
  Tau_profile_exit_all_threads();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(clear_doc, "\
clear()\n\
\n\
Clear all profiling information collected so far.\n\
");

static PyObject *profiler_clear(ProfilerObject *pObj, PyObject* noarg) {
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(getPythonCompileVersion_doc, "\
getPythonCompileVersion()\n\
\n\
Get the version of Python that the TAU C extension was compiled against.\n\
");

static PyObject *profiler_getPythonCompileVersion(ProfilerObject *pObj, PyObject* noarg) {
  int major = PY_MAJOR_VERSION;
  int minor = PY_MINOR_VERSION;
  int micro = PY_MICRO_VERSION;
  PyObject * result = Py_BuildValue("iii", major, minor, micro);
  return result;
}

static void profiler_dealloc(ProfilerObject *op) {
  if (op->flags & POF_ENABLED)
    PyEval_SetProfile(NULL, NULL);
  flush_unmatched(op);
  clearEntries(op);
  Py_XDECREF(op->externalTimer);
  Py_TYPE(op)->tp_free(op);
}

static int profiler_init(ProfilerObject *pObj, PyObject *args, PyObject *kw) {
  PyObject *o;
  PyObject *timer = NULL;
  double timeunit = 0.0;
  int subcalls = 1;
#ifdef PyTrace_C_CALL
  int builtins = 1;
#else
  int builtins = 0;
#endif
  static char *kwlist[] = {"timer", "timeunit",
			   "subcalls", "builtins", 0};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "|Odii:Profiler", kwlist,
				   &timer, &timeunit,
				   &subcalls, &builtins))
    return -1;

  if (setSubcalls(pObj, subcalls) < 0 || setBuiltins(pObj, builtins) < 0)
    return -1;
  o = pObj->externalTimer;
  pObj->externalTimer = timer;
  Py_XINCREF(timer);
  Py_XDECREF(o);
  pObj->externalTimerUnit = timeunit;
  return 0;
}

static PyMethodDef profiler_methods[] = {
  {"getstats",    (PyCFunction)profiler_getstats,
   METH_NOARGS,			getstats_doc},
  {"enable",	(PyCFunction)profiler_enable,
   METH_VARARGS | METH_KEYWORDS,	enable_doc},
  {"disable",	(PyCFunction)profiler_disable,
   METH_NOARGS,			disable_doc},
  {"clear",	(PyCFunction)profiler_clear,
   METH_NOARGS,			clear_doc},
  {"exitAllThreads",	(PyCFunction)profiler_exitAllThreads,
   METH_NOARGS,			exitAllThreads_doc},
  {"getPythonCompileVersion",	(PyCFunction)profiler_getPythonCompileVersion,
   METH_STATIC | METH_NOARGS,  getPythonCompileVersion_doc},
  {NULL, NULL}
};

PyDoc_STRVAR(profiler_doc, "\
Profiler(custom_timer=None, time_unit=None, subcalls=True, builtins=True)\n\
\n\
    Builds a profiler object using the specified timer function.\n\
    The default timer is a fast built-in one based on real time.\n\
    For custom timer functions returning integers, time_unit can\n\
    be a float specifying a scale (i.e. how long each integer unit\n\
    is, in seconds).\n\
");

statichere PyTypeObject PyProfiler_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "ctau_impl.Profiler",                     /* tp_name */
  sizeof(ProfilerObject),                 /* tp_basicsize */
  0,                                      /* tp_itemsize */
  (destructor)profiler_dealloc,           /* tp_dealloc */
  0,                                      /* tp_print */
  0,                                      /* tp_getattr */
  0,                                      /* tp_setattr */
  0,                                      /* tp_compare */
  0,                                      /* tp_repr */
  0,                                      /* tp_as_number */
  0,                                      /* tp_as_sequence */
  0,                                      /* tp_as_mapping */
  0,                                      /* tp_hash */
  0,                                      /* tp_call */
  0,                                      /* tp_str */
  0,                                      /* tp_getattro */
  0,                                      /* tp_setattro */
  0,                                      /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  profiler_doc,                           /* tp_doc */
  0,                                      /* tp_traverse */
  0,                                      /* tp_clear */
  0,                                      /* tp_richcompare */
  0,                                      /* tp_weaklistoffset */
  0,                                      /* tp_iter */
  0,                                      /* tp_iternext */
  profiler_methods,                       /* tp_methods */
  0,                                      /* tp_members */
  0,                                      /* tp_getset */
  0,                                      /* tp_base */
  0,                                      /* tp_dict */
  0,                                      /* tp_descr_get */
  0,                                      /* tp_descr_set */
  0,                                      /* tp_dictoffset */
  (initproc)profiler_init,                /* tp_init */
  PyType_GenericAlloc,                    /* tp_alloc */
  PyType_GenericNew,                      /* tp_new */
  PyObject_Del,                           /* tp_free */
};

static PyMethodDef moduleMethods[] = {
  {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef ctau_impl_moduledef = {
            PyModuleDef_HEAD_INIT,
            "ctau_impl",         /* m_name */
            "TAU Fast Profiler", /* m_doc */
            -1,                  /* m_size */
            moduleMethods,       /* m_methods */
            NULL,                /* m_reload */
            NULL,                /* m_traverse */
            NULL,                /* m_clear */
            NULL,                /* m_free */
        };
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_ctau_impl(void)
#else
initctau_impl(void)
#endif
{
  PyObject *module, *d;
#if PY_MAJOR_VERSION >= 3
  module = PyModule_Create(&ctau_impl_moduledef);
#else
  module = Py_InitModule3("ctau_impl", moduleMethods, "TAU Fast profiler");
#endif
  if (module == NULL)
#if defined(__APPLE__) && PY_MAJOR_VERSION < 3
    return ;
#else
    return NULL;
#endif /* __APPLE__ */
  d = PyModule_GetDict(module);
  if (PyType_Ready(&PyProfiler_Type) < 0)
#if defined(__APPLE__) && PY_MAJOR_VERSION < 3
    return ;
#else
    return NULL;
#endif /* __APPLE__ */
  PyDict_SetItemString(d, "Profiler", (PyObject *)&PyProfiler_Type);

  if (!initialized) {
    PyStructSequence_InitType(&StatsEntryType,
			      &profiler_entry_desc);
    PyStructSequence_InitType(&StatsSubEntryType,
			      &profiler_subentry_desc);
  }
  Py_INCREF((PyObject*) &StatsEntryType);
  Py_INCREF((PyObject*) &StatsSubEntryType);
  PyModule_AddObject(module, "profiler_entry",
		     (PyObject*) &StatsEntryType);
  PyModule_AddObject(module, "profiler_subentry",
		     (PyObject*) &StatsSubEntryType);
  empty_tuple = PyTuple_New(0);
  initialized = 1;
#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}
