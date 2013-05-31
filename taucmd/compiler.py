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
from taucmd import configuration
from taucmd import TauNotImplementedError

LOGGER = taucmd.getLogger(__name__)


# Tau compiler wrapper scripts
TAU_CC = 'tau_cc.sh'
TAU_CXX = 'tau_cxx.sh'
TAU_F77 = 'tau_f90.sh'     # Yes, f90 not f77
TAU_F90 = 'tau_f90.sh'
TAU_UPC = 'tau_upc.sh'

# List of all compiler classes
ALL_COMPILERS = list()


class Compiler(object):
    """
    Base class for all compiler classes
    """
    
    @classmethod
    def create(cls, family, tag, language, tau_cmd, commands):
        newcc = type('%s_%s' % (family.TAG, tag), (Compiler,),
                     dict(FAMILY=family,
                          NAME='%s %s Compiler' % (family.NAME, language),
                          TAG=tag,
                          LANGUAGE=language,
                          TAU_COMMAND=tau_cmd,
                          COMMANDS=commands))
        ALL_COMPILERS.append(newcc)
        return newcc


class Family(object):
    """
    Base class for all compiler families
    """
    
    @classmethod
    def add(cls, tag, language, tau_cmd, commands):
        setattr(cls, tag, Compiler.create(cls, tag, language, tau_cmd, commands))

    @classmethod
    def populate(cls, **kwargs):
        compilers = {'CC': ('C', TAU_CC),
                     'CXX': ('C++', TAU_CXX),
                     'F77': ('FORTRAN77', TAU_F77),
                     'F90': ('Fortran90', TAU_F90),
                     'UPC': ('Universal Parallel C', TAU_UPC)}
        for tag, commands in kwargs.iteritems():
            cls.add(tag, compilers[tag][0], compilers[tag][1], commands)
    
    @classmethod
    def create(cls, tag, name, **kwargs):
        newfamily = type('%s_Family' % tag, (Family,), dict(NAME=name, TAG=tag))
        newfamily.populate(**kwargs)
        return newfamily

GNU_COMPILERS = Family.create('GNU', 'GNU Compiler Collection', 
                              CC=['gcc'], CXX=['g++'], F77=['gfortran'], F90=['gfortran'], UPC=['gupc'])

MPI_COMPILERS = Family.create('MPI', 'MPI Compiler Wrappers', 
                              CC=['mpicc'], CXX=['mpicxx', 'mpic++'], F77=['mpif77'], F90=['mpif90'])


def known_compiler_commands():
    """
    Returns the known compiler commands
    """
    known = list()
    for cc in ALL_COMPILERS:
        for cmd in cc.COMMANDS:
            if not cmd in known:
                known.append(cmd)
                yield cmd

def identify(cmd):
    """
    Looks up the compliler class for a given compiler command
    """
    for cc in ALL_COMPILERS:
        if cmd in cc.COMMANDS:
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
    LOGGER.info('Recognized %r as compiler command (%s)' % (cmd, cc.NAME))
    
    # Load the configuration registry
    registry = configuration.Registry.load()
    if not len(registry):
        print "No Tau configurations have been created.  Use the 'config' subcommand to create a configuration."
        print "See tau config --help for more info."
        return 1
    
    # Load a configuration
    try:
        config = registry.loadDefault()
    except KeyError:
        print 'There is no configuration named %r at %r' % (taucmd.CONFIG, registry.prefix)
        print 'Valid names are:'
        for name in registry:
            print name
        print "Use the --config argument to specify the configuration name or the 'config' subcommand to create a new configuration."
        print "See tau --help for more info."
        return 1
    LOGGER.info('Selected configuration %r' % taucmd.CONFIG)
    LOGGER.debug('Configuration details:\n%s' % config)
    
    # Build the configuration
    compiled = config.build(cc)
    
    # Record in the registry that the configuration is built
    registry.save()
    
    # Invoke the TAU compiler wrapper script
    cmd = [cc.TAU_COMMAND] + cmd_args
    env = compiled.getEnvironment()
    LOGGER.debug('Creating subprocess: cmd=%r, env=%r' % (cmd, env))
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    return proc.wait()
