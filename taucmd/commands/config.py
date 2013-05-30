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
import logging
import taucmd
from docopt import docopt
from taucmd import configuration

USAGE = """
Usage:
  tau config [options]

Configuration Options:
  --name=<name>                     Configuration name. [default: %(name_default)s]
  --default=<name>                  Set the default configuration name.
  --delete=<name>                   Delete a configuration.
  --list                            Show all configurations.

Assisting Library Options:
  --pdt=(download|<path>)           Program Database Toolkit (PDT) installation path. [default: download]
  --bfd=(download|<path>)           GNU Binutils installation path. [default: download]
  --dyninst=(download|<path>)       DyninstAPI installation path. [default: download]
  --papi=<path>                     Performance API (PAPI) installation path.
  
Multithreading Options:
  --threads=(openmp|pthread)        Select multithreading library.
                                    openmp: Use the compiler's OpenMP libraries.
                                    pthread: Use POSIX threads.

Message Passing Interface (MPI) Options:
  --mpi=<path>                      MPI installation path.
  --mpi-include=<path>              MPI header files installation path.
  --mpi-lib=<path>                  MPI library files installation path.

NVIDIA CUDA Options:
  --cuda=<path>                     NVIDIA CUDA SDK installation path.
  --cuda-libs=<libraries>           Additional linker options used when linking to the CUDA driver libraries.     

Universal Parallel C (UPC) Options:
  --upc-gasnet=<path>               GASNET installation path.
  --upc-network=(auto|mpi|smp)      Specify UPC network.
"""

SHORT_DESCRIPTION = "Create and manage Tau configurations."

HELP = """
Help page to be written.
"""

COMMANDS = ['--default', '--delete', '--list']

def get_usage():
    """
    Returns a string describing subcommand usage
    """
    return USAGE % {'name_default':  taucmd.CONFIG}

def main(argv):
    """
    Program entry point
    """

    # Parse command line arguments
    args = docopt(get_usage(), argv=argv)
    logging.debug('Arguments: %s' % args)

    # Load configuration registry
    registry = configuration.Registry.load(taucmd.HOME, taucmd.CONFIG)

    # Check for --list command
    if args['--list']:
        print registry
        return 0

    # Check for --set-default command
    name = args['--default']
    if name:
        try:
            registry.set_default(name)
            registry.save()
            return 0
        except KeyError:
            print 'There is no configuration named %r at %r' % (name, registry.prefix)
            print 'Valid names are:'
            for name in registry:
                print name
            return 1

    # Check for --delete command
    name = args['--delete']
    if name:
        try:
            registry.unregister(name)
            registry.save()
            return 0
        except KeyError:
            print 'There is no configuration named %r at %r' % (name, registry.prefix)
            print 'Valid names are:'
            for name in registry:
                print name
            return 1

    # Translate command line arguments to configuration data
    config = dict()
    for key, val in args.iteritems():
        if key[0:2] == '--' and not key in COMMANDS:
            config[key[2:]] = val

    # Add configuration to registry
    try:
        registry.register(config)
        registry.save()
        return 0
    except KeyError:
        print 'A configuration named %r already exists.' % config['name']
        print 'Use the --name option to specify a different name, '
        print 'or use the --delete option to remove the existing configuration.'
        print 'See tau config --help for more information.'
        return 1
    
        