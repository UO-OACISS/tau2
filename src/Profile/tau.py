#! /usr/bin/env python
#
# Class for profiling python code. rev 1.0  6/2/94
#
# Based on prior profile module by Sjoerd Mullender...
#   which was hacked somewhat by: Guido van Rossum
#
# See profile.doc for more information

"""Class for profiling Python code."""

# Copyright 1994, by InfoSeek Corporation, all rights reserved.
# Written by James Roskind
#
# Permission to use, copy, modify, and distribute this Python software
# and its associated documentation for any purpose (subject to the
# restriction in the following sentence) without fee is hereby granted,
# provided that the above copyright notice appears in all copies, and
# that both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of InfoSeek not be used in
# advertising or publicity pertaining to distribution of the software
# without specific, written prior permission.  This permission is
# explicitly restricted to the copying and modification of the software
# to remain in Python, compiled Python, or other languages (such as C)
# wherein the modified or derived code is exclusively imported into a
# Python module.
#
# INFOSEEK CORPORATION DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
# SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL INFOSEEK CORPORATION BE LIABLE FOR ANY
# SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
# CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.



import sys
import os
import time
import marshal
import pytau
from optparse import OptionParser

__all__ = ["run", "runctx", "help", "Profile"]


#**************************************************************************
# The following are the static member functions for the profiler class
# Note that an instance of Profile() is *not* needed to call them.
#**************************************************************************


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
    prof = Profile()
    try:
        prof = prof.run(statement)
    except SystemExit:
        pass
    pytau.stop()

def runctx(statement, globals, locals, filename=None):
    """Run statement under profiler, supplying your own globals and locals,
    optionally saving results in filename.

    statement and filename have the same semantics as profile.run
    """
    prof = Profile()
    try:
        prof = prof.runctx(statement, globals, locals)
    except SystemExit:
        pass
    pytau.stop()



# print help
def help():
    for dirname in sys.path:
        fullname = os.path.join(dirname, 'profile.doc')
        if os.path.exists(fullname):
            sts = os.system('${PAGER-more} ' + fullname)
            if sts: print '*** Pager exit status:', sts
            break
    else:
        print 'Sorry, can\'t find the help file "profile.doc"',
        print 'along the Python search path.'



class Profile:

    def __init__(self, timer=None):
        self.c_func_name = ""
        self.dispatcher = self.trace_dispatch

    def trace_dispatch(self, frame, event, arg):
        if event == "c_call":
            self.c_func_name = arg.__name__
        self.dispatch[event](self, frame)

    def trace_dispatch_exception(self, frame):
        return 1

    def trace_dispatch_call(self, frame):
        fcode = frame.f_code

        classname = ""
        if frame.f_locals:
            obj = frame.f_locals.get("self", None)
            if not obj is None:
                classname = obj.__class__.__name__ + "::"
            else:
                classname = ""

        methodname = fcode.co_name
        # methods with "?" are usually the files themselves (no method)
        # we now name them based on the file
        if methodname == "?":
            methodname = fcode.co_filename
            methodname = methodname[methodname.rfind("/")+1:]
        tauname = classname + methodname
        filename = fcode.co_filename[fcode.co_filename.rfind("/")+1:]
        tautype = '[{' + filename + '}{' + str(fcode.co_firstlineno) + '}]'

        # exclude the "? <string>" timer
        if not fcode.co_filename == "<string>":
            tautimer = pytau.profileTimer(tauname, tautype)
            pytau.start(tautimer)

    def trace_dispatch_return(self, frame):
        # exclude the "? <string>" timer
        if not frame.f_code.co_filename == "<string>":
            pytau.stop()


    def trace_dispatch_c_call (self, frame):
        if self.c_func_name == "start" or self.c_func_name == "stop" or self.c_func_name == "profileTimer" or self.c_func_name == "setprofile":
            pass
        else:
            tautimer = pytau.profileTimer(self.c_func_name, "")
            pytau.start(tautimer)


    def trace_dispatch_c_return(self, frame):
        if self.c_func_name == "start" or self.c_func_name == "stop" or self.c_func_name == "profileTimer":
            pass
        else:
            pytau.stop()


    dispatch = {
        "call": trace_dispatch_call,
        "exception": trace_dispatch_exception,
        "return": trace_dispatch_return,
        "c_call": trace_dispatch_c_call,
        "c_exception": trace_dispatch_return,  # the C function returned
        "c_return": trace_dispatch_c_return,
        }


    class fake_code:
        def __init__(self, filename, line, name):
            self.co_filename = filename
            self.co_line = line
            self.co_name = name
            self.co_firstlineno = 0

        def __repr__(self):
            return repr((self.co_filename, self.co_line, self.co_name))

    class fake_frame:
        def __init__(self, code, prior, local):
            self.f_code = code
            self.f_back = prior
            self.f_locals = local

    def simulate_call(self, name):
        code = self.fake_code('profile', 0, name)
        frame = self.fake_frame(code, None, None)
        self.dispatch['call'](self, frame)



    # The following two methods can be called by clients to use
    # a profiler to profile a statement, given as a string.

    def run(self, cmd):
        import __main__
        dict = __main__.__dict__
        return self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals, locals):
        self.simulate_call(cmd)
        sys.setprofile(self.dispatcher)
        try:
            exec cmd in globals, locals
        finally:
            sys.setprofile(None)
        return self

    # This method is more useful to profile a single function call.
    def runcall(self, func, *args, **kw):
        self.simulate_call(repr(func))
        sys.setprofile(self.dispatcher)
        try:
            return func(*args, **kw)
        finally:
            sys.setprofile(None)



# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    usage = "tau.py scriptfile [arg] ..."
    if not sys.argv[1:]:
        print "Usage: ", usage
        sys.exit(2)

    class ProfileParser(OptionParser):
        def __init__(self, usage):
            OptionParser.__init__(self)
            self.usage = usage

    parser = ProfileParser(usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
        help="Save stats to <outfile>", default=None)
    parser.add_option('-s', '--sort', dest="sort",
        help="Sort order when printing to stdout, based on pstats.Stats class", default=-1)

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    if (len(sys.argv) > 0):
        sys.path.insert(0, os.path.dirname(sys.argv[0]))
        run('execfile(%r)' % (sys.argv[0],), options.outfile, options.sort)
    else:
        print "Usage: ", usage
