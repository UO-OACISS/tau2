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
import errno
import logging
import ConfigParser
from docopt import docopt

USAGE = """
Usage:
  tau config [options]

Installation Options:
  --prefix=<path>                   Tau installation path. [default: %(prefix_default)s]
  --arch=<arch>                     Target architecture. [default: %(arch_default)s]
  --shared=(yes|no)                 Enable/disable Tau shared library. [default: yes]

Assisting Library Options:
  --pdt=(auto|download|<path>)      Program Database Toolkit (PDT) installation path. [default: auto]
  --bfd=(auto|download|<path>)      GNU Binutils installation path. [default: download]
  --dyninst=(auto|download|<path>)  DyninstAPI installation path. [default: auto]
  --papi=(auto|<path>)              Performance API (PAPI) installation path. [default: auto]
  --jdk=(auto|<path>)               Java Development Toolkit (JDK) installation path. [default: auto]
  --scorep=<path>                   Score-P installation path.

Multithreading:
  --threads=<library>               Set threading library.  One of:
                                      pthreads: POSIX threads library
                                      openmp:   Compiler's OpenMP library

Message Passing Interface (MPI) Options:
  --mpi=<path>                      MPI installation path. [default: auto]
  --mpi-include=<path>              MPI header files installation path.
  --mpi-lib=<path>                  MPI library files installation path.
  --mpi-track-comm=(yes|no)         Enable/disable communication event tracking. [default: yes]

NVIDIA CUDA Options:
  --cuda=(auto|<path>)              NVIDIA CUDA SDK installation path. [default: auto]
  --cuda-libs=<libraries>           Additional linker options used when linking to the CUDA driver libraries [default: None]     

Universal Parallel C (UPC) Options:
  --upc-gasnet=<path>               GASNET installation path.
  --upc-network=(auto|mpi|smp)      Specify UPC network. [default: auto]
"""

SHORT_DESCRIPTION = "Create a new Tau configuration file."

HELP = """
Help page to be written.
"""

def detect_host_arch():
    """
    Tries to autodetect the host architecture.
    """
    
    # For now, just for x86_64
    return 'x86_64'

def detect_host_compiler():
    """
    Tries to autodetect the host compiler suite.
    """
    
    # Just GNU for now
    return 'gnu'


def main(argv):
    """
    Program entry point
    """

    # Get some default values
    prefix_default = os.path.join(os.path.expanduser('~'), '.tau')
    arch_default = detect_host_arch()
    compiler_default = detect_host_compiler()
    
    # Parse command line arguments
    usage = USAGE % {'prefix_default': prefix_default,
                     'arch_default': arch_default,
                     'compiler_default': compiler_default}
    args = docopt(usage, argv=argv)
    logging.debug('Arguments: %s' % args)
    
    # Translate command line arguments to configuration file
    config = ConfigParser.SafeConfigParser()
    default_section = 'tau'
    config.add_section(default_section)
    for key, val in args.iteritems():
        if key[0:2] == '--':
            config.set(default_section, key[2:], str(val))

    # Create installation prefix
    try:
        os.makedirs(args['--prefix'])
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(args['--prefix']): pass
        else: raise

    # Save configuration file
    config_file = os.path.join(args['--prefix'], 'config')
    with open(config_file, 'wb') as f:
        config.write(f)
