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
from docopt import docopt
from hashlib import md5
from taucmd import configuration, compiler, TauConfigurationError
from taucmd.registry import Registry

LOGGER = taucmd.getLogger(__name__)

USAGE = """
Usage:
  tau config new [<name>] --compiler=<family> [options]
  tau config default <name>
  tau config delete <name>
  tau config list
  tau config --help
  
Subcommands:
  new                               Create a new configuration. 
  default                           Set the default configuration.
  delete                            Delete a configuration.
  list                              List all configurations.

Compiler Options:
  --compiler=<family>               Set the compiler family.  One of:
                                    %(families)s

Target Options:
  --target=<name>                   Set target. [default: %(target_default)s]

Assisting Library Options:
  --pdt=<path>                      Program Database Toolkit (PDT) installation path.
  --bfd=(download|<path>)           GNU Binutils installation path. [default: download]
  --dyninst=<path>                  DyninstAPI installation path.
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

Universal Parallel C (UPC) Options:
  --upc-gasnet=<path>               GASNET installation path.
  --upc-network=(auto|mpi|smp)      Specify UPC network.
"""

SHORT_DESCRIPTION = "Manage Tau configurations."

HELP = """
Help page to be written.
"""

def detectTarget():
    """
    Use TAU's archfind script to detect the target architecture
    """
    cmd = os.path.join(taucmd.TAU_ROOT_DIR, 'utils', 'archfind')
    return subprocess.check_output(cmd).strip()

def getConfigurationName(args):
    """
    Builds a name for the configuration
    """
    if args['<name>']:
        return args['<name>']
    else:
        nameparts = ['--pdt', '--bfd', '--dyninst', '--papi', '--mpi', '--cuda']
        valueparts = ['--compiler', '--target', '--threads']
        parts = [args[part].lower() for part in valueparts if args[part]]
        parts.extend([part[2:].lower() for part in nameparts if args[part]])
        return '-'.join(parts)

def getConfiguration(args):
    """
    Create a TauConfiguration object from command line arguments
    """
    path_args = ['--pdt', '--bfd', '--dyninst', '--papi', '--mpi', 
                 '--mpi-include', '--mpi-lib', '--cuda', '--upc-gasnet']
    
    # Check for invalid compiler families
    family_tags = [family.tag for family in compiler.ALL_FAMILIES]
    if not args['--compiler'].upper() in family_tags:
        msg = 'Unrecognized compiler family: %r' % args['--compiler']
        hint = 'Known families: %s' % ', '.join(family_tags)
        raise TauConfigurationError(msg, hint)

    # Check for invalid paths
    for arg in path_args:
        path = args[arg]
        if path and path != 'download':
            path = args[arg] = os.path.abspath(os.path.expanduser(path))
            if not (os.path.exists(path) and os.path.isdir(path)):
                raise TauConfigurationError("Invalid argument: %s=%s: '%s' does not exist or is not a directory." % (arg, path, path),
                                            'Check the command arguments and try again.')

    # Populate configuration data
    config = configuration.TauConfiguration() 
    config['name'] = getConfigurationName(args)
    for key, val in args.iteritems():
        if key[0:2] == '--':
            config[key[2:]] = val

    # Calculate configuration ID
    hash = md5()
    for item in sorted(config.data.iteritems()):
        hash.update(repr(item))
    config['id'] = hash.hexdigest()
    config['prefix'] = os.path.join(taucmd.HOME, config['id'])
    return config
    

def getUsage():
    """
    Returns a string describing subcommand usage
    """
    parts = list()
    for family in compiler.ALL_FAMILIES:
        parts.append('  %s  (%s)' % ('{:<6}'.format(family.tag), family.name))
    families = ('\n'+' '*36).join(parts)
    return USAGE % {'families': families,
                    'target_default': detectTarget()}

def main(argv):
    """
    Program entry point
    """

    # Parse command line arguments
    args = docopt(getUsage(), argv=argv)
    if args['--compiler']:
        args['--compiler'] = args['--compiler'].upper()
    LOGGER.debug('Arguments: %s' % args)

    # Load configuration registry
    registry = Registry.load()

    # Check for --list command
    if args['list']:
        print registry
        return 0

    # Check for --set-default command
    if args['default']:
        name = args['<name>']
        try:
            registry.setDefault(name)
            registry.save()
            return 0
        except KeyError:
            msg = 'There is no configuration named %r at %r' % (name, registry.prefix)
            if len(registry):
                hint = 'Valid names are: %s' % ', '.join([name for name in registry])
            else:
                hint = 'No configurations have been defined.'
            raise TauConfigurationError(msg, hint)

    # Check for --delete command
    if args['delete']:
        name = args['<name>']
        try:
            registry.unregister(name)
            registry.save()
            return 0
        except KeyError:
            msg = 'There is no configuration named %r at %r' % (name, registry.prefix)
            if len(registry):
                hint = 'Valid names are: %s' % ', '.join([name for name in registry])
            else:
                hint = 'No configurations have been defined.'
            raise TauConfigurationError(msg, hint)

    # Check for --new command
    if args['new']:
        # Translate command line arguments to configuration data
        config = getConfiguration(args)
        # Add configuration to registry
        try:
            registry.register(config)
            registry.save()
            return 0
        except KeyError:
            msg = 'A configuration named %r already exists.' % config['name']
            hint = "Use 'tau config new <name>' name the configuration. See tau config --help for more info."
            raise TauConfigurationError(msg, hint)

    # Don't know what you want so show usage
    #print getUsage()
        