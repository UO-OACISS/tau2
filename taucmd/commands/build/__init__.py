"""
@file
@author John C. Linford (jlinford@paratools.com)
@version 1.0

@brief

This file is part of the TAU Performance System

@section COPYRIGHT

Copyright (c) 2013, ParaTools, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
 (1) Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
 (2) Redistributions in binary form must reproduce the above copyright notice, 
     this list of conditions and the following disclaimer in the documentation 
     and/or other materials provided with the distribution.
 (3) Neither the name of ParaTools, Inc. nor the names of its contributors may 
     be used to endorse or promote products derived from this software without 
     specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import sys
import subprocess
import taucmd
from taucmd import commands
from taucmd.docopt import docopt
from taucmd.project import Registry
from pkgutil import walk_packages
from textwrap import dedent

LOGGER = taucmd.getLogger(__name__)

SHORT_DESCRIPTION = "Instrument programs during compilation and/or linking."

USAGE = """
Usage:
  tau build <compiler> [<args>...]
  tau build -h | --help
  
Known Compiler Commands:
%(simple_descr)s
%(command_descr)s
"""

HELP = """
'tau build' help page to be written.
"""

# Tau compiler wrapper scripts
TAU_COMPILERS = {'CC': 'tau_cc.sh',
                 'CXX': 'tau_cxx.sh',
                 'F77': 'tau_f77.sh',
                 'F90': 'tau_f90.sh',
                 'UPC': 'tau_upc.sh'}

class SimpleCompiler(object):
    def __init__(self, cmd, lang, descr):
        self.cmd = cmd
        self.tau_cmd = TAU_COMPILERS[lang]
        self.short_descr = '%s Compiler.' % descr
        self.usage = 'Usage:\n  tau build %s <args>...' % cmd
        self.help = 'Invokes the TAU compiler wrapper script %r for compilation with the %s compiler.' % (self.tau_cmd, descr)

SIMPLE_COMPILERS = {'cc': SimpleCompiler('cc', 'CC', 'C'),
                    'c++': SimpleCompiler('c++', 'CXX', 'C++'),
                    'f77': SimpleCompiler('f77', 'F77', 'FORTRAN77'),
                    'f90': SimpleCompiler('f90', 'F90', 'Fortran90'),
                    'ftn': SimpleCompiler('ftn', 'F90', 'Fortran90'),
                    'gcc': SimpleCompiler('gcc', 'CC', 'GNU C'),
                    'g++': SimpleCompiler('g++', 'CXX', 'GNU C++'),
                    'gfortran': SimpleCompiler('gfortran', 'F90', 'GNU Fortran90'),
                    'icc': SimpleCompiler('icc', 'CC', 'Intel C'),
                    'icpc': SimpleCompiler('icpc', 'CXX', 'Intel C++'),
                    'ifort': SimpleCompiler('ifort', 'F90', 'Intel Fortran90'),
                    'pgcc': SimpleCompiler('pgcc', 'CC', 'Portland Group C'),
                    'pgCC': SimpleCompiler('pgCC', 'CXX', 'Portland Group C++'),
                    'pgf77': SimpleCompiler('pgf77', 'F77', 'Portland Group FORTRAN77'),
                    'pgf90': SimpleCompiler('ptf90', 'F90', 'Portland Group Fortran90'),
                    'mpicc': SimpleCompiler('mpicc', 'CC', 'MPI C'),
                    'mpicxx': SimpleCompiler('mpicxx', 'CXX', 'MPI C++'),
                    'mpic++': SimpleCompiler('mpic++', 'CXX', 'MPI C++'),
                    'mpiCC': SimpleCompiler('mpiCC', 'CXX', 'MPI C++'),
                    'mpif77': SimpleCompiler('mpif77', 'F77', 'MPI FORTRAN77'),
                    'mpif90': SimpleCompiler('mpif90', 'F90', 'MPI Fortran90')}

def getUsage():
    parts = ['  %s  %s' % ('{:<15}'.format(comp.cmd), comp.short_descr) for comp in SIMPLE_COMPILERS.itervalues()]
    parts.sort()
    return USAGE % {'simple_descr': '\n'.join(parts), 
                    'command_descr': commands.getSubcommands(__name__)}

def getHelp():
    return HELP

def isKnownCompiler(cmd):
    """
    Returns True if cmd is a known compiler command
    """
    known = SIMPLE_COMPILERS.keys() + [n for _, n, _ in walk_packages(sys.modules[__name__].__path__)]
    return cmd in known

def simpleCompile(compiler, argv):
    LOGGER.debug('Arguments: %r' % argv)
    cmd_args = argv[2:]
    
    # Get selected project
    registry = Registry()
    proj = registry.getSelectedProject()
    if not proj:
        print "There are no TAU projects in %r.  See 'tau project create'." % os.getcwd()
        return 1
    print 'Using TAU project %r' % proj.getName()

    # Check project compatibility
    if not proj.supportsCompiler(compiler):
        print '!'*80
        print '!'
        print "! Warning: %r project may not support the %r compiler command." % (proj.getName(), compiler)
        if proj.hasCompilers():
            print "! Supported compilers: %r" % proj.getCompilers()
        print '!'
        print '!'*80
    
    # Compile the project if needed
    proj.compile()

    # Set the environment
    env = proj.getTauCompilerEnvironment()
    
    # Get compiler flags
    flags = proj.getTauCompilerFlags()
    
    # Execute the compiler wrapper script
    if cmd_args:
        cmd = [SIMPLE_COMPILERS[compiler].tau_cmd] + flags + cmd_args
    else:
        cmd = [SIMPLE_COMPILERS[compiler].cmd]

    LOGGER.debug('Creating subprocess: cmd=%r, env=%r' % (cmd, env))
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    return proc.wait()


def main(argv):
    """
    Program entry point
    """

    # Parse command line arguments
    usage = getUsage()
    args = docopt(usage, argv=argv, options_first=True)
    LOGGER.debug('Arguments: %s' % args)
    cmd = args['<compiler>']
    cmd_args = args['<args>']
    
    # Check for -h | --help (why doesn't this work automatically?)
    idx = cmd.find('-h')
    if (idx == 0) or (idx == 1):
        print usage
        return 0
    
    # Execute the simple compilation if supported
    if cmd in SIMPLE_COMPILERS:
        return simpleCompile(cmd, argv)

    # Try to execute as a tau command
    cmd_module = 'taucmd.commands.build.%s' % cmd
    try:
        __import__(cmd_module)
        LOGGER.debug('Recognized %r as a tau build command' % cmd)
        return sys.modules[cmd_module].main(['build', cmd] + cmd_args)
    except ImportError:
        LOGGER.debug('%r not recognized as a tau build command' % cmd)
        print usage
        return 1
