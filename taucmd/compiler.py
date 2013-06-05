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
import inspect
from taucmd import TauNotImplementedError, TauConfigurationError
from taucmd import configuration
from taucmd.registry import Registry

LOGGER = taucmd.getLogger(__name__)


# Tau compiler wrapper scripts
TAU_CC = 'tau_cc.sh'
TAU_CXX = 'tau_cxx.sh'
TAU_F77 = 'tau_f90.sh'     # Yes, f90 not f77
TAU_F90 = 'tau_f90.sh'
TAU_UPC = 'tau_upc.sh'

# Map compiler tags to Tau compiler wrapper scripts
TAU_COMPILERS = {'CC':  ('C', TAU_CC),
                 'CXX': ('C++', TAU_CXX),
                 'F77': ('FORTRAN77', TAU_F77),
                 'F90': ('Fortran90', TAU_F90),
                 'UPC': ('Universal Parallel C', TAU_UPC)}

ALL_COMPILERS = set()
ALL_FAMILIES = set()

class Compiler(object):
    """
    A compiler
    """
    def __init__(self, tag, family, language, tau_cmd, commands):
        self.tag = tag
        self.family = family
        self.language = language
        self.tau_command = tau_cmd
        self.commands = commands
        self.name = '%s %s Compiler' % (family.name, language)
        ALL_COMPILERS.add(self)


class Family(object):
    """
    A compiler family
    """
    def __init__(self, tag, name, **kwargs):
        self.tag = tag
        self.name = name
        for tag, commands in kwargs.iteritems():
            setattr(self, tag, Compiler(tag, self, TAU_COMPILERS[tag][0], 
                                        TAU_COMPILERS[tag][1], commands))
        ALL_FAMILIES.add(self)


GNU_COMPILERS = Family('GNU', 'GNU Compiler Collection', 
                       CC=['gcc'], CXX=['g++'], F77=['gfortran'], 
                       F90=['gfortran'], UPC=['gupc'])

MPI_COMPILERS = Family('MPI', 'MPI Compiler Wrappers', 
                       CC=['mpicc'], CXX=['mpicxx', 'mpic++'], 
                       F77=['mpif77'], F90=['mpif90'])


def knownCompilerCommands():
    """
    Returns the known compiler commands
    """
    known = list()
    for cc in ALL_COMPILERS:
        for cmd in cc.commands:
            if not cmd in known:
                known.append(cmd)
                yield cmd

def identify(cmd):
    """
    Looks up the compliler class for a given compiler command
    """
    for cc in ALL_COMPILERS:
        if cmd in cc.commands:
            return cc
        
def compile(args):
    """
    Loads a TAU configuration and launches the compiler wrapper script.
    """
    cmd = args['<command>']
    cmd_args = args['<args>']
    cc = identify(cmd)
    if not cc:
        raise TauNotImplementedError("%r: unknown compiler command. Try 'tau --help'." % cmd, cmd)
    LOGGER.info('Recognized %r as %s' % (cmd, cc.name))
    
    # Load the configuration registry
    registry = Registry.load()
    if not len(registry):
        raise TauConfigurationError("No Tau configurations have been created.",
                                    "Use the 'config' subcommand to create a configuration.")

    # Load a configuration
    try:
        config = registry[taucmd.CONFIG]
    except KeyError:
        message = 'There is no configuration named %r at %r' % (taucmd.CONFIG, registry.prefix)
        hint = 'Valid names are: %s.' % ', '.join([name for name in registry])
        raise TauConfigurationError(message, hint)
    LOGGER.info('Selected configuration %r' % taucmd.CONFIG)
    LOGGER.debug('Configuration details:\n%s' % config)
    
    # Build the configuration for use with a compiler
    config.build(cc)

    # Record in the registry that the configuration is built
    registry.save()
    
    # Invoke the command
    if cmd_args:
        cmd = [cc.tau_command] + cmd_args
    env = config.getEnvironment()
    LOGGER.debug('Creating subprocess: cmd=%r, env=%r' % (cmd, env))
#    if taucmd.LOG_LEVEL == 'DEBUG':
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
#    else:
#        with open(os.devnull, 'w') as devnull:
#            proc = subprocess.Popen(cmd, env=env, stdout=devnull, stderr=devnull)
    return proc.wait()
