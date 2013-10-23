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
import glob
import subprocess
import fnmatch
import taucmd
import shutil
from taucmd import TauError, TauNotImplementedError

LOGGER = taucmd.getLogger(__name__)

# User specific TAU source code location
TAU_SRC_DIR = os.path.join(taucmd.SRC_DIR, 'tau')



def cloneSource(dest=TAU_SRC_DIR, source=taucmd.TAU_MASTER_SRC_DIR):
    """
    Makes a fresh clone of the TAU source code
    """
    # Don't copy if the source already exists
    if os.path.exists(dest) and os.path.isdir(dest):
        LOGGER.debug('TAU source code directory %r already exists.' % dest)
        return
    # Filename filter for copytree
    def ignore(path, names):
        globs = ['*.o', '*.a', '*.so', '*.dylib', '*.pyc', 'a.out', 
                 '.all_configs', '.last_config', '.project', '.cproject',
                 '.git', '.gitignore', '.ptp-sync', '.pydevproject']
        # Ignore bindirs in the top level directory
        if path == source:
            bindirs = ['x86_64', 'bgl', 'bgp', 'bgq', 'craycnl', 'apple']
            globs.extend(bindirs)
        # Build set of ignored files
        ignored_names = []
        for pattern in globs:
            ignored_names.extend(fnmatch.filter(names, pattern))
        return set(ignored_names)

    LOGGER.debug('Copying from %r to %r' % (source, dest))
    LOGGER.info('Creating new copy of TAU at %r.  This will only be done once.' % dest)
    shutil.copytree(source, dest, ignore=ignore)


def translateConfigureArg(config, key, val):
    """
    Gets the configure script argument(s) corresponding to a Tau Commander argument
    """
    # Ignore unspecified arguments
    if not val:
        return []
    # No parameter flags
    noparam = {'mpi': '-mpi',
               'openmp': '-openmp',
               'pthreads': '-pthread',
               'pdt': '-pdt=%s' % config['pdt-prefix'],
               'bfd': '-bfd=%s' % config['bfd-prefix'],
               'cuda': '-cuda=%s' % config['cuda-sdk']}
    # One parameter flags
    oneparam = {'dyninst': '-dyninst=%s',
                'mpi-include': '-mpiinc=%s',
                'mpi-lib': '-mpilib=%s',
                'papi': '-papi=%s',
                'tau-prefix': '-prefix=%s',
                'target-arch': '-arch=%s',
                'upc': '-upc=%s',
                'upc-gasnet': '-gasnet=%s',
                'upc-network': '-upcnetwork=%s',
                #TODO: Translate compiler command correctly
                'cc': '-cc=%s',
                'c++': '-c++=%s',
                'fortran': '-fortran=%s',
                'pdt_c++': '-pdt_c++=%s'}
    # Attempt no-argument translation
    try:
        return [noparam[key]]
    except KeyError:
        pass
    # Attempt one-argument translation
    try:
        return [oneparam[key] % val]
    except KeyError:
        pass
    # Couldn't translate the argument
    return []


def getConfigureCommand(config):
    """
    Returns the command that will configure TAU for this project
    """
    cmd = ['./configure']
    for key, val in config.iteritems():
        cmd.extend(translateConfigureArg(config, key, val))
    return cmd


def getPrefix(config):
    nameparts = ['bfd', 'cuda', 'dyninst', 'mpi', 'openmp', 'papi', 'pdt', 'pthreads']
    valueparts = ['c++', 'cc', 'fortran', 'target-arch', 'upc', 'upc-network']
    parts = [config[part].lower() for part in valueparts if config[part]]
    parts.extend([part.lower() for part in nameparts if config[part]])
    parts.sort()
    name = '_'.join(parts)
    prefix = os.path.join(taucmd.TAUCMD_HOME, 'tau', name)
    return prefix


def install(config, stdout=sys.stdout, stderr=sys.stderr):
    """
    Installs TAU
    """
    prefix = getPrefix(config)
    if os.path.isdir(prefix):
        LOGGER.debug("Skipping TAU installation.  %r is a directory." % prefix)
        return
    
    # Banner
    print 'Installing TAU at %r' % prefix

    # Clone the TAU source code to the user's home directory
    cloneSource()

    # Configure the source code for this configuration
    srcdir = TAU_SRC_DIR
    cmd = getConfigureCommand(config)
    LOGGER.debug('Creating configure subprocess in %r: %r' % (srcdir, cmd))
    print 'Configuring TAU...'
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('TAU configure failed.')
    
    # Execute make
    cmd = ['make', '-j', 'install']
    LOGGER.debug('Creating make subprocess in %r: %r' % (srcdir, cmd))
    print 'Compiling TAU...'
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('TAU compilation failed.')
    
    # Leave source, we'll probably need it again soon
    print 'TAU installation complete.'
        
