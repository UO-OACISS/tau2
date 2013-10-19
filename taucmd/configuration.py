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
import fnmatch
import glob
import taucmd
from shutil import copytree, rmtree
from pprint import pformat
from taucmd import TauError

LOGGER = taucmd.getLogger(__name__)

# Default settings
DEFAULT_TAU_OPTIONS = ['-optRevert']

def translateConfigureArg(key, val):
    """
    Gets the configure script argument(s) corresponding to a Tau Commander argument
    """
    # Ignore empty arguments
    if not val: 
        return []

    # Ignore some arguments
    if key in ['name', 'id', 'compiler']:
        return []

    # Simple translations
    simple = {'bfd': '-bfd=%s',
              'cuda': '-cuda=%s',
              'dyninst': '-dyninst=%s',
              'mpi-include': '-mpiinc=%s',
              'mpi-lib': '-mpilib=%s',
              'papi': '-papi=%s',
              'pdt': '-pdt=%s',
              'prefix': '-prefix=%s',
              'target': '-arch=%s',
              'upc-gasnet': '-gasnet=%s',
              'upc-network': '-upcnetwork=%s'}
    
    # Map of multithread options
    threadmap = {'openmp': '-openmp',
                 'pthread': '-pthread'}
    
    # Attempt a simple translation
    try:
        return [simple[key] % val]
    except KeyError:
        pass
    
    # Can't do a simple translation, do a more complex translation
    if key == 'mpi':
        mpiinc = os.path.join(val, 'include')
        mpilib = os.path.join(val, 'lib')
        if not (os.path.exists(mpilib) and os.path.isdir(mpilib)):
            mpilib = os.path.join(val, 'lib64') 
        return ['-mpiinc=%s' % mpiinc, '-mpilib=%s' % mpilib]
    elif key == 'threads':
        return [threadmap[val]]

    # Couldn't translate the argument
    raise TauError('Cannot translate configuration parameter %r' % key)



def cloneTauSource(dest, source=taucmd.TAU_ROOT_DIR):
    """
    Makes a fresh clone of the TAU source code
    """
    # Don't copy if the source already exists
    if os.path.exists(dest) and os.path.isdir(dest):
        LOGGER.debug('TAU source code directory %r already exists.' % dest)
        return

    # Filename filter for copytree
    def ignore(path, names):
        # Globs to ignore 
        patterns = ['*.o', '*.a', '*.so', '*.dylib', '*.pyc', 'a.out', 
                 '.all_configs', '.last_config', '.project', '.cproject',
                 '.git', '.gitignore', '.ptp-sync', '.pydevproject']
        # Ignore bindirs in the top level directory
        if path == taucmd.TAU_ROOT_DIR:
            bindirs = ['x86_64', 'bgl', 'bgp', 'bgq', 'craycnl', 'apple']
            patterns.extend(bindirs)
        # Build set of ignored files
        ignored_names = []
        for pattern in patterns:
            ignored_names.extend(fnmatch.filter(names, pattern))
        return set(ignored_names)

    LOGGER.debug('Copying from %r to %r and ignoring %r' % (source, dest, ignore))
    LOGGER.info('Creating new copy of TAU at %r.  This will only be done once.' % dest)
    copytree(source, dest, ignore=ignore)


class TauConfiguration(object):
    """
    A Tau configuration
    """
    def __init__(self):
        self.data = dict()
        self.built = False
        
    def __str__(self):
        try:
            return '%s:\n%s' % (self['name'], pformat(self.data))
        except KeyError:
            return '(empty)'
        
    def __len__(self):
        return self.data.__len__()
    
    def __iter__(self):
        return self.data.__iter__()
    
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    
    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)
    
    def __contains__(self, item):
        return self.data.__contains__(item)
    
    def _getConfigureCommand(self, family):
        """
        Returns the command that will configure TAU for the given configuration
        """
        cmd = ['./configure']
        if family.CC:
            cmd.append('-cc=%s' % family.CC.commands[0])
        if family.CXX:
            cmd.append('-c++=%s' % family.CXX.commands[0])
        if family.F77 or family.F90:
            cmd.append('-fortran=%s' % family.F90.commands[0])
        for key, val in self.data.iteritems():
            cmd.extend(translateConfigureArg(key, val))
        return cmd

    def build(self, cc):
        """
        Builds the configuration.
        """
        
        # Don't build the configuration if it's already built
        if self.built:
            LOGGER.debug('Configuration %r is already built.' % self.data['name'])
            return self
        LOGGER.debug('Configuration will be installed at %r' % self.data['prefix'])

        # Prepare the TAU source code
        srcdir = os.path.join(taucmd.HOME, 'src')
        cloneTauSource(srcdir)

        # Choose to display configure/build output
        devnull = None
        if taucmd.LOG_LEVEL == 'DEBUG':
            stdout = sys.stdout
            stderr = sys.stderr
        else:
            devnull = open(os.devnull, 'w')
            stdout = devnull
            stderr = devnull

        LOGGER.info('Building TAU configuration %r for use with %r. This will only be done once.' % (self.data['name'], cc.family.name))

        # Configure the source code for this configuration
        cmd = self._getConfigureCommand(cc.family)
        LOGGER.debug('Creating configure subprocess in %r: %r' % (srcdir, cmd))
        proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
        if proc.wait():
            rmtree(self.data['prefix'], ignore_errors=True)
            raise TauError('TAU configure failed.')
        
        # Execute make
        cmd = ['make', '-j', 'install']
        LOGGER.debug('Creating make subprocess in %r: %r' % (srcdir, cmd))
        proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
        if proc.wait():
            rmtree(self.data['prefix'], ignore_errors=True)
            raise TauError('TAU compilation failed.')
        
        # Mark this configuration as built
        if devnull:
            devnull.close() 
        self.built = True
    
    def getMakefile(self):
        """
        Returns TAU_MAKEFILE for this configuration
        """
        makefiles = os.path.join(self.data['prefix'], self.data['target'], 'lib', 'Makefile.tau*')
        return glob.glob(makefiles)[0]

    def getEnvironment(self):
        """
        Returns an environment for use with subprocess.Popen that specifies the
        TAU-specific environment variables for this configuration
        """
        env = os.environ
        env['TAU_OPTIONS'] = ' '.join(DEFAULT_TAU_OPTIONS)
        env['TAU_MAKEFILE'] = self.getMakefile()
        bindir = os.path.join(self.data['prefix'], self.data['target'], 'bin')
        try:
            env['PATH'] = bindir + ':' + env['PATH']
            LOGGER.debug('Updated PATH to %r' % env['PATH'])
        except KeyError:
            LOGGER.warning('The PATH environment variable was unset.')
            env['PATH'] = bindir
        return env