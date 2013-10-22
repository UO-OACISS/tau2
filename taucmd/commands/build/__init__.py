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

LOGGER = taucmd.getLogger(__name__)

SHORT_DESCRIPTION = "Instrument programs during compilation and/or linking."

USAGE = """
Usage:
  tau build <compiler> [<args>...]
  tau build -h | --help
  
Known Compilers:
%(command_descr)s
"""

HELP = """
'tau build' help page to be written.
"""

# Tau compiler wrapper scripts
TAU_CC = 'tau_cc.sh'
TAU_CXX = 'tau_cxx.sh'
TAU_F77 = 'tau_f77.sh'
TAU_F90 = 'tau_f90.sh'
TAU_UPC = 'tau_upc.sh'

SIMPLE_COMPILERS = {'gcc': TAU_CC,
                    'g++': TAU_CXX,
                    'gfortran': TAU_F90,
                    'mpicc': TAU_CC,
                    'mpicxx': TAU_CXX,
                    'mpic++': TAU_CXX,
                    'mpif77': TAU_F77,
                    'mpif90': TAU_F90}

def getUsage():
    return USAGE % {'command_descr': commands.getSubcommands(__name__)}

def getHelp():
    return HELP

def isKnownCompiler(cmd):
    """
    Returns True if cmd is a known compiler command
    """
    known = [n for _, n, _ in walk_packages(sys.modules[__name__].__path__)]
    return cmd in known

def simpleCompile(compiler, argv):
    LOGGER.debug('Arguments: %r' % argv)
    cmd_args = argv[2:]
    if not cmd_args:
        print "ERROR: no options specified"
        return 1
    
    registry = Registry()
    if not len(registry.projects):
        print "There are no TAU projects in %r.  See 'tau project create'." % os.getcwd()
        return 1

    # Check project compatibility
    proj = registry.getSelectedProject()
    print 'Using TAU project %r' % proj.getName()
    if not proj.supportsCompiler(compiler):
        print "Warning: %r project may not support the %r compiler command.  Supported compilers: %r" % (proj.getName(), compiler, proj.getCompilers())
    
    # Compile the project if needed
    proj.compile()

    # Set the environment
    env = proj.getTauCompilerEnvironment()
    
    # Get compiler flags
    flags = proj.getTauCompilerFlags()
    
    # Execute the compiler wrapper script
    cmd = [SIMPLE_COMPILERS[compiler]] + flags + cmd_args

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
    cmd_module = 'taucmd.commands.build.%s' % cmd   
    
    # Check for -h | --help (why doesn't this work automatically?)
    idx = cmd.find('-h')
    if (idx == 0) or (idx == 1):
        print usage
        return 0
    
    # Execute the simple compilation if supported
    if cmd in SIMPLE_COMPILERS:
        return simpleCompile(cmd, argv)

    # Try to execute as a tau command
    
    try:
        __import__(cmd_module)
        LOGGER.debug('Recognized %r as a tau build command' % cmd)
        return sys.modules[cmd_module].main(['build', cmd] + cmd_args)
    except ImportError:
        LOGGER.debug('%r not recognized as a tau build command' % cmd)
        print usage
        return 1
