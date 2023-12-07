#! /usr/bin/env python

"""Python interface for TAU."""

__all__ = ["run", "runctx", "exitAllThreads", "help", "Profile"]

import ctau_impl
import pytau

# ____________________________________________________________
# Simple interface

def get_ctau_python_version():
    return ctau_impl.Profiler.getPythonCompileVersion()

def writeProfiles(prefix="profile"):
    import pytau
    pytau.dbDump(prefix)

def taupy_write_trace(in_trace):
    linenum = 0
    while in_trace:
        metadata_name = "PY-BACKTRACE( ) {0}".format(linenum)
        metadata_val = str(in_trace.pop())
        pytau.metadata(metadata_name, metadata_val)
        linenum = linenum + 1


import signal, sys, traceback,os
def taupy_signal_handler(signum, frame):
    ext_trace = []
    for threadId, stack in sys._current_frames().items():
        for filename, lineno, funcname, line in traceback.extract_stack(stack):
            if funcname == sys._getframe().f_code.co_name:
                continue
            ext_trace.append("[{0}] [{1}:{2}]".format(funcname, filename, lineno))
    #Signals do not have the same value in different OS systems
    #and strsignal(signalnum) is not available until v3.8
    metadata_name = "PY-SIGNAL"
    if signum == signal.SIGILL:
        metadata_val = "Illegal Instruction"
    if signum == signal.SIGINT:
        metadava_val = "Keyboard Interruption"
    if signum == signal.SIGQUIT:
        metadava_val = "Quit signal"
    if signum == signal.SIGTERM:
        metadata_val = "Termination signal"
    if signum == signal.SIGPIPE:
        metadata_val = "Broken pipe"
    elif signum == signal.SIGABRT:
        metadata_val = "Abort signal"
    elif signum == signal.SIGFPE:
        metadata_val = "Floating-point exception"
    elif signum == signal.SIGBUS:
        metadata_val = "Bus error(bad memory access)"
    elif signum == signal.SIGSEGV :
        metadata_val = "Segmentation Fault"
    else:
        metadata_val = "Unknown signal"
    pytau.metadata(metadata_name, metadata_val)
    taupy_write_trace(ext_trace)
    sys.exit(1)
            
def taupy_excepthook(type, value, tb):
    ext_trace = []
    for filename, lineno, funcname, line in traceback.extract_tb(tb):
        ext_trace.append("[{0}] [{1}:{2}]".format(funcname, filename, lineno))
    metadata_name = "PY-Exception"
    metadata_val = str(value)
    pytau.metadata(metadata_name, metadata_val)
    taupy_write_trace(ext_trace)

def taupy_listen_signals():
    if os.getenv("TAU_TRACK_PYSIGNALS")=='1':
        signal.signal(signal.SIGILL,  taupy_signal_handler)
        signal.signal(signal.SIGINT,  taupy_signal_handler)
        signal.signal(signal.SIGQUIT, taupy_signal_handler)
        signal.signal(signal.SIGTERM, taupy_signal_handler)
        signal.signal(signal.SIGPIPE, taupy_signal_handler)
        signal.signal(signal.SIGABRT, taupy_signal_handler)
        signal.signal(signal.SIGFPE,  taupy_signal_handler)
        #TauEnv_get_memdbg
        signal.signal(signal.SIGBUS,  taupy_signal_handler)
        signal.signal(signal.SIGSEGV, taupy_signal_handler)

def taupy_enable_excepthook():
    if os.getenv("TAU_TRACK_PYSIGNALS")=='1':
        sys.excepthook = taupy_excepthook

def run(statement, filename=None, sort=-1):
    """Run statement under profiler optionally saving results in filename

    This function takes a single argument that can be passed to the
    "exec" statement, and an optional file name.  In all cases this
    routine attempts to "exec" its first argument and gather profiling
    statistics from the execution. If no file name is present, then this
    function automatically prints a simple profiling report, sorted by the
    standard name string (file/line/function-name) that is presented in
    each line.
    """

    taupy_listen_signals()
    taupy_enable_excepthook()
    prof = Profile()
    result = None
    try:
        prof = prof.run(statement)
    except SystemExit:
        pass
    finally:
        if filename:
            prof.dump_stats(filename)
        else:
            result = prof.print_stats(sort)
    return result

def runctx(statement, globals, locals, filename=None):
    """Run statement under profiler, supplying your own globals and locals,
    optionally saving results in filename.

    statement and filename have the same semantics as profile.run
    """
    taupy_listen_signals()
    taupy_enable_excepthook()
    prof = Profile()
    result = None
    try:
        try:
            prof = prof.runctx(statement, globals, locals)
        except SystemExit:
            pass
    finally:
        if filename is not None:
            prof.dump_stats(filename)
        else:
            result = prof.print_stats()
    return result

def runmodule(modname, filename=None):
    """
    Compile and run a module specified by 'modname', setting __main__ to that module.
    """
    taupy_listen_signals()
    taupy_enable_excepthook()
    prof = Profile()
    result = None
    try:
        try:
            prof = prof.runmodule(modname)
        except SystemExit:
            pass
    finally:
        if filename is not None:
            prof.dump_stats(filename)
        else:
            result = prof.print_stats()
    return result

def runmoduledir(modname, filename=None):
    """
    Compile and run a module directory specified by 'modnamedir', setting __main__ to that module.
    """
    taupy_listen_signals()
    taupy_enable_excepthook()
    prof = Profile()
    result = None
    try:
        try:
            prof = prof.runmoduledir(modname)
        except SystemExit:
            pass
    finally:
        if filename is not None:
            prof.dump_stats(filename)
        else:
            result = prof.print_stats()
    return result

def exitAllThreads():
    Profile().exitAllThreads()

# Backwards compatibility.
def help():
    print("Documentation for the profile/cProfile modules can be found ")
    print("in the Python Library Reference, section 'The Python Profiler'.")

# ____________________________________________________________

class Profile(ctau_impl.Profiler):
    """Profile(custom_timer=None, time_unit=None, subcalls=True, builtins=True)

    Builds a profiler object using the specified timer function.
    The default timer is a fast built-in one based on real time.
    For custom timer functions returning integers, time_unit can
    be a float specifying a scale (i.e. how long each integer unit
    is, in seconds).
    """

    # Most of the functionality is in the base class.
    # This subclass only adds convenient and backward-compatible methods.

    def print_stats(self, sort=-1):
        import pstats
#        pstats.Stats(self).strip_dirs().sort_stats(sort).print_stats()

    def dump_stats(self, file):
        import marshal
        f = open(file, 'wb')
        self.create_stats()
        marshal.dump(self.stats, f)
        f.close()

    def create_stats(self):
        self.disable()
        self.snapshot_stats()

    def snapshot_stats(self):
        entries = self.getstats()
        self.stats = {}
        callersdicts = {}
        # call information
        for entry in entries:
            func = label(entry.code)
            nc = entry.callcount         # ncalls column of pstats (before '/')
            cc = nc - entry.reccallcount # ncalls column of pstats (after '/')
            tt = entry.inlinetime        # tottime column of pstats
            ct = entry.totaltime         # cumtime column of pstats
            callers = {}
            callersdicts[id(entry.code)] = callers
            self.stats[func] = cc, nc, tt, ct, callers
        # subcall information
        for entry in entries:
            if entry.calls:
                func = label(entry.code)
                for subentry in entry.calls:
                    try:
                        callers = callersdicts[id(subentry.code)]
                    except KeyError:
                        continue
                    nc = subentry.callcount
                    cc = nc - subentry.reccallcount
                    tt = subentry.inlinetime
                    ct = subentry.totaltime
                    if func in callers:
                        prev = callers[func]
                        nc += prev[0]
                        cc += prev[1]
                        tt += prev[2]
                        ct += prev[3]
                    callers[func] = nc, cc, tt, ct

    # The following two methods can be called by clients to use
    # a profiler to profile a statement, given as a string.

    def run(self, cmd):
        import __main__
        dict = __main__.__dict__
        return self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals, locals):
        self.enable()
        try:
            import pytau
            x = pytau.profileTimer(cmd)
            pytau.start(x)
            exec(cmd, globals, locals)
            pytau.stop(x)
        finally:
            self.disable()
        return self

    # Python 3.12 and later remove the `imp` module.
    # We have to use `runpy` instead.
    def runmodule_runpy(self, modname, newname='__main__'):
        self.enable()
        try:
            import runpy
            runpy.run_module(modname, run_name=newname, alter_sys=True)
        finally:
            self.disable()
            return self

    def runmodule(self, modname, newname='__main__'):
        import sys
        try:
            from imp import find_module, new_module, PY_SOURCE
        except ImportError as e:
            return self.runmodule_runpy(modname, newname)
        replaced = sys.modules.get(newname, None)
        self.enable()
        try:
            fileobj, path, desc = find_module(modname)
            if desc[2] != PY_SOURCE:
                raise ImportError('No source file found for module %s' % modname)
            code = compile(fileobj.read(), path, 'exec')
            module = new_module(newname)
            if '__file__' not in module.__dict__:
                module.__dict__['__file__'] = path
            sys.modules[newname] = module
            exec(code, module.__dict__)
        finally:
            if replaced:
                sys.modules[newname] = replaced
            self.disable()
        return self

    def runmoduledir(self, modname, newname='__main__'):
        self.enable()
        try:
            import runpy
            runpy.run_module(modname, run_name='__main__', alter_sys=True)
        finally:
            self.disable()
            return self

    # This method is more useful to profile a single function call.
    def runcall(self, func, *args, **kw):
        self.enable()
        try:
            return func(*args, **kw)
        finally:
            self.disable()

    def runcall_with_node(self, node, func, *args, **kw):
        self.enable()
        try:
            import pytau
            pytau.setNode(node)
            return func(*args, **kw)
        finally:
            self.disable()

try:
    from pyspark import BasicProfiler
    try:
        from pyspark import TaskContext
    except Exception:
        print("WARNING: Your version of Spark does not expose TaskContext to Python. " \
              "PySpark tasks will not be profiled.")
    from pyspark import TaskContext
    class TauSparkProfiler(BasicProfiler):
        
        def __init__(self, ctx):
            BasicProfiler.__init__(self, ctx)

        def profile(self, func):
            myId = TaskContext._getOrCreate()._taskAttemptId + 1 # Leave 0 for the driver
            prof = Profile()
            prof.runcall_with_node(myId, func)
            writeProfiles()
except Exception:
    # Do nothing if Spark isn't available
    pass


# ____________________________________________________________

def label(code):
    if isinstance(code, str):
        return ('~', 0, code)    # built-in functions ('~' sorts at the end)
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)

# ____________________________________________________________

def main():
    import os, sys
    from optparse import OptionParser
    usage = "cProfile.py [-o output_file_path] [-s sort] scriptfile [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
        help="Save stats to <outfile>", default=None)
    parser.add_option('-s', '--sort', dest="sort",
        help="Sort order when printing to stdout, based on pstats.Stats class", default=-1)

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    if (len(sys.argv) > 0):
        sys.path.insert(0, os.path.dirname(sys.argv[0]))
        run('execfile(%r)' % (sys.argv[0],), options.outfile, options.sort)
    else:
        parser.print_usage()
    return parser

# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    main()
