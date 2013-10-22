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
from datetime import datetime
from taucmd.project import Registry, ProjectNameError, isProjectNameValid
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
  --cc=<compiler>                   Set C compiler.
  --c++=<compiler>                  Set C++ compiler.  
  --fortran=<compiler>              Set Fortran compiler.
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
  --memory                          Enable memory instrumentation.
  --memory-debug                    Enable memory debugging.

I/O and Communication Options:
  --comm-matrix                     Build the application's communication matrix.
  --io                              Enable I/O instrumentation.

Callpath Options:
  --callpath=<number>               Set the callpath depth in the application profile. [default: 0]
  
Measurement Options:
  --profile                         Enable application profiling.
  --trace                           Enable application tracing.
  --sample                          Enable application sampling.  Requires --bfd.
"""

HELP = """
'project create' page to be written.
"""

# # Compilers used when no other compiler specified
# DEFAULT_COMPILERS = [('--cc', 'gcc'), ('--c++', 'g++'), ('--fortran', 'gfortran')]
# 
# # Compilers used when MPI specified
# MPI_COMPILERS = [('--cc', 'mpicc'), ('--c++', 'mpicxx'), ('--fortran', 'mpif90')]

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
    
    # Make sure at least one measurement method is used
    if not (args['--profile'] or args['--trace'] or args['--sample']):
        args['--profile'] = True 
        
#     # Set compilers if not set
#     if args['--mpi']:
#         compilers = MPI_COMPILERS
#     else:
#         compilers = DEFAULT_COMPILERS
#     for flag, comp in compilers:
#         if not args[flag]:
#             args[flag] = comp 

    # Strip and check args
    config = {'refresh': True,
              'modified': datetime.now()}
    exclude = ['--help', '-h', '--select']
    for key, val in args.iteritems():
        if key[0:2] == '--' and not key in exclude:
            if key == '--name' and val and isProjectNameValid(val):
                print "Error: %r cannot be used as a project name.  See 'tau project select --help'." % val
                return 1
            elif key in ['--pdt', '--bfd']:
                if val.upper() == 'NONE':
                    config[key[2:]] = None
                elif val.upper() == 'DOWNLOAD':
                    config[key[2:]] = 'download'
                else:
                    config[key[2:]] = val
            else:
                config[key[2:]] = val

    # TODO: Other PDT compilers
    config['pdt_c++'] = 'g++'

    registry = Registry()
    try:
        proj = registry.addProject(config, args['--select'])
    except ProjectNameError, e:
        print e.value
        return 1
    
    proj_name = proj.getName()
    select_name = registry.getSelectedProject().getName()
    if proj:
        print 'Created project %r' % proj_name
        if select_name == proj_name:
            print 'Selected %r as the new default project' % select_name
        else:
            print "Note: The selected project is %r.\n      Type 'tau project select %s' to select this project." % (select_name, proj_name)
        return 0
