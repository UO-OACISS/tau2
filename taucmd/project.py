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
import errno
import fnmatch
import taucmd
import pickle
from shutil import copytree

LOGGER = taucmd.getLogger(__name__)


def mkdirp(path):
    """
    Creates a directory and all its parents.
    """
    try:
        os.makedirs(path)
        LOGGER.debug('Created directory %r' % path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path): pass
        else: raise


def cloneTauSource(dest=os.path.join(taucmd.TAU_HOME, 'src'), source=taucmd.TAU_ROOT_DIR):
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
        if path == taucmd.TAU_ROOT_DIR:
            bindirs = ['x86_64', 'bgl', 'bgp', 'bgq', 'craycnl', 'apple']
            globs.extend(bindirs)
        # Build set of ignored files
        ignored_names = []
        for pattern in globs:
            ignored_names.extend(fnmatch.filter(names, pattern))
        return set(ignored_names)

    LOGGER.debug('Copying from %r to %r' % (source, dest))
    LOGGER.info('Creating new copy of TAU at %r.  This will only be done once.' % dest)
    copytree(source, dest, ignore=ignore)


def loadProjects(path='.tau'):
    try:
        with open(os.path.join(path, 'projects'), 'rb') as fp:
            return pickle.load(fp)
    except:
        return {}
    
    
def saveProjects(projects, path='.tau'):
    mkdirp(path)
    with open(os.path.join(path, 'projects'), 'wb') as fp:
        pickle.dump(projects, fp)


def initialize(config):
    # Store paths in project configuration
    project_home = os.path.join('.tau', getName(config))
    config['project_home'] = project_home
    # Clone the TAU source code to the user's home directory
    cloneTauSource()
    # Prepare the project directory
    mkdirp(project_home)


def getName(config):
    if config['name']:
        return config['name']
    else:
        nameparts = ['bfd', 'binary-inst', 'callpath', 'comm-matrix', 'compiler-inst', 
                     'cuda', 'dyninst', 'io', 'memory', 'memory-debug', 'mpi', 'openmp',
                     'papi', 'pdt', 'profile', 'pthreads', 'sample', 'source-inst', 'trace']
        valueparts = ['c++', 'cc', 'fortran', 'target-arch', 'upc', 'upc-network']
        parts = [config[part].lower() for part in valueparts if config[part]]
        parts.extend([part.lower() for part in nameparts if config[part]])
        name = '_'.join(parts)
        config['name'] = name
        return name

