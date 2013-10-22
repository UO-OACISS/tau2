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
import subprocess
import taucmd
from taucmd.project import Registry, ProjectNameError
from taucmd.docopt import docopt

LOGGER = taucmd.getLogger(__name__)

SHORT_DESCRIPTION = "Create a new TAU project configuration."

USAGE = """
Usage:
  tau project create [options]
  tau project create -h | --help

Project Options:
  --name=<name>                     Set project name.
  --select                          Select this project after creating it.
  
Architecture Options:
  --target-arch=<name>              Set target architecture. [default: %(target_default)s]

Compiler Options:
  --cc=<compiler>                   Set C compiler. [default: gcc]
  --c++=<compiler>                  Set C++ compiler. [default: g++]  
  --fortran=<compiler>              Set Fortran compiler. [default: gfortran]
  --upc=<compiler>                  Set UPC compiler.

Assisting Library Options:
  --pdt=(download|<path>|none)      Program Database Toolkit (PDT) installation path. [default: download]
  --bfd=(download|<path>|none)      GNU Binutils installation path. [default: download]
  --dyninst=<path>                  DyninstAPI installation path.
  --papi=<path>                     Performance API (PAPI) installation path.
  
Thread Options:
  --openmp                          Enable OpenMP instrumentation.
  --pthreads                        Enable pthreads instrumentation.
  
Message Passing Interface (MPI) Options:
  --mpi                             Enable MPI instrumentation.
  --mpi-include=<path>              MPI header files installation path.
  --mpi-lib=<path>                  MPI library files installation path.

NVIDIA CUDA Options:
  --cuda                            Enable NVIDIA CUDA instrumentation.
  --cuda-sdk=<path>                 NVIDIA CUDA SDK installation directory. [default: /usr/local/cuda]

Universal Parallel C (UPC) Options:
  --upc-gasnet=<path>               GASNET installation path.
  --upc-network=<network>           Set UPC network.

Memory Options:
  --memory                          Enable memory instrumentation. [default: False]
  --memory-debug                    Enable memory debugging. [default: False]

I/O and Communication Options:
  --comm-matrix                     Build the application's communication matrix. [default: False]
  --io                              Enable I/O instrumentation. [default: False]

Callpath Options:
  --callpath                        Show callpaths in application profile. [default: False]
  --callpath-depth=<number>         Set the depth of callpaths in the application profile. [default: 25]

Instrumentation Options:
  --source-inst                     Enable source instrumentation. Requires --pdt. [default: True]
  --compiler-inst                   Enable compiler instrumentation. [default: False]
  --binary-inst                     Enable binary rewriting instrumentation.  [default: False]
  
Measurement Options:
  --profile                         Enable application profiling. [default: True]
  --trace                           Enable application tracing. [default: False]
  --sample                          Enable application sampling.  Requires --bfd. [default: False]
  --profile-format=(packed|merged)  Set profile file format. [default: packed]
  --trace-format=(slog2|otf)        Set trace file format. [default: slog2] 
"""

HELP = """
'project create' page to be written.
"""

def getUsage():
    return USAGE % {'target_default': detectTarget()}

def getHelp():
    return HELP


def detectTarget():
    """
    Use TAU's archfind script to detect the target architecture
    """
    cmd = os.path.join(taucmd.TAU_MASTER_SRC_DIR, 'utils', 'archfind')
    return subprocess.check_output(cmd).strip()


def main(argv):
    """
    Program entry point
    """
    # Parse command line arguments
    usage = getUsage()
    args = docopt(usage, argv=argv)
    LOGGER.debug('Arguments: %s' % args)

    registry = Registry()
    try:
        proj = registry.newProject(args)
    except ProjectNameError, e:
        print e.value
        return 1
    
    proj_name = proj.getName()
    default_name = registry.getDefaultProject().getName()
    if proj:
        print 'Created project %r' % proj_name
        if default_name == proj_name:
            print 'Selected %r as the new default project' % default_name
        else:
            print "Note: The selected project is %r.\n      Type 'tau project select %s' to select this project." % (default_name, proj_name)
        return 0
