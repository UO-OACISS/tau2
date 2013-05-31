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
import pickle
import errno
import subprocess
import fnmatch
import glob
import taucmd
from shutil import rmtree, copytree, ignore_patterns
from pprint import pformat
from hashlib import md5
from taucmd import TAU_ROOT_DIR
from taucmd import TauError

LOGGER = taucmd.getLogger(__name__)

# Default settings
TAU_OPTIONS = ['-optRevert']



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

def clone_tau_source(dest):
    """
    Makes a fresh clone of the TAU source code
    """
    source = TAU_ROOT_DIR
    
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
        if path == TAU_ROOT_DIR:
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


class Registry(object):
    """
    The registry of Tau configurations
    """
    def __init__(self, registry_file, prefix, default):
        self._registry_file = registry_file
        self.prefix = prefix
        self.data = dict()
        self.default = default

    def __str__(self):
        sep = '-'*80
        if len(self.data):
            parts = [sep]
            for id, config in self.data.iteritems():
                parts.append(str(config))
                parts.append(sep)
            parts.append('Default configuration: %s' % self.default)
            parts.append(sep)
            return '\n'.join(parts)
        else:
            return sep + '\nNo configurations\n' + sep
    
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
    
    @classmethod
    def load(cls, prefix=taucmd.HOME, default=taucmd.CONFIG):
        """
        Loads the configuration registry from file.
        """
        registry_file = os.path.join(prefix, 'registry')
        if os.path.exists(registry_file):
            with open(registry_file, 'rb') as f:
                registry = pickle.load(f)
                registry._registry_file = registry_file
                LOGGER.debug('Registry loaded from file %r' % registry_file)
                return registry
        else:
            LOGGER.debug('Registry file %r does not exist.' % registry_file)
            return cls(registry_file, prefix,  default)

    def save(self):
        """
        Saves the configuration registry to file.
        """
        mkdirp(self.prefix)
        with open(self._registry_file, 'wb') as f:
            pickle.dump(self, f)
            LOGGER.debug('Wrote registry file %r' % self._registry_file)

    def register(self, config):
        """
        Adds a new configuration to the registry
        """
        name = config['name']
        # Check for conflicting configurations
        if name in self.data:
            raise KeyError
        # Calculate configuration ID
        hash = md5()
        for item in sorted(config.data.iteritems()):
            if item[0] != 'id':
                hash.update(repr(item))
        config['id'] = hash.hexdigest()
        # Register configuration
        self.data[name] = config
        LOGGER.info('Registered new configuration %r' % name)
        # Set this configuration to default if it's the only configuration
        if len(self.data) == 1:
            self.setDefault(name)

    def unregister(self, name):
        """
        Removes a configuration from the registry
        """
        id = self.data[name].data['id']
        # Remove from registry
        del self.data[name]
        LOGGER.info('Unregistered configuration %r' % name)
        # Remove old files
        LOGGER.info('Deleting configuration files')
        for path in glob.glob(os.path.join(taucmd.HOME, '*', id)):
            LOGGER.debug('Deleting %r' % path)
            shutil.rmtree(path, ignore_errors=True)
        # Change default if we just deleted the default configuration
        if name == self.default and len(self.data):
            self.setDefault(self.data.values()[0]['name'])

    def setDefault(self, name):
        """
        Records the name of the default configuration in the registry
        """
        if name in self.data:
            self.default = name
            LOGGER.info('Set default configuration to %r' % self.default)
        else:
            raise KeyError
        
    def loadDefault(self):
        """
        Returns the default configuration
        """
        return self.data[self.default]

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
    
    def translate_configure_arg(self, key, val):
        """
        Gets the configure script argument(s) corresponding to a Tau Commander argument
        """
        # Ignore empty arguments
        if not val: 
            return ''

        # Ignore some arguments
        if key in ['name', 'id']:
            return ''
        
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
            return simple[key] % val
        except KeyError:
            pass
        
        # Can't do a simple translation, do a more complex translation
        if key == 'mpi':
            mpiinc = os.path.join(val, 'include')
            mpilib = os.path.join(val, 'lib')
            if not (os.path.exists(mpilib) and os.path.isdir(mpilib)):
                mpilib = os.path.join(val, 'lib64') 
            return '-mpiinc=%s -mpilib=%s' % (mpiinc, mpilib)
        elif key == 'threads':
            return threadmap[val]

        # Couldn't translate the argument
        raise TauError('Cannot translate configuration parameter %r' % key)


    def _get_configure_command(self):
        """
        Returns the command that will configure TAU for the given configuration
        """
        cmd = ['./configure', '-prefix=%s' % self.prefix]
        if self.family.CC:
            cmd.append('-cc=%s' % self.family.CC.COMMANDS[0])
        if self.family.CXX:
            cmd.append('-c++=%s' % self.family.CXX.COMMANDS[0])
        if self.family.F77 or self.family.F90:
            cmd.append('-fortran=%s' % self.family.F90.COMMANDS[0])
        for key, val in self.data.iteritems():
            translation = self.translate_configure_arg(key, val)
            if translation: 
                cmd.append(translation)
        return cmd

    def _compiled_by_family(self, family):
        """
        Make a copy of this configuration specified for a compiler family
        """
        cfg = TauConfiguration()
        cfg.__dict__.update(self.__dict__)
        cfg.family = family
        cfg.prefix = os.path.join(taucmd.HOME, os.path.join(family.TAG, cfg.data['id']))
        LOGGER.info('Building configuration %r for use with %r' % (cfg.data['name'], family.NAME))
        return cfg

    def build(self, cc):
        """
        Builds the configuration.
        """
        
        # Only build the configuration once
        compiled = self._compiled_by_family(cc.FAMILY)
        if compiled.built:
            LOGGER.debug('Configuration %r is already built.' % compiled.data['name'])
            return compiled
        LOGGER.debug('Configuration will be installed at %r' % compiled.prefix)
        LOGGER.info('The %r configuration will be compiled.  This will only be done once.' % compiled.data['name'])

        # Prepare the TAU source code
        srcdir = os.path.join(taucmd.HOME, 'src')
        clone_tau_source(srcdir)
        
        # Configure the source code for this configuration
        cmd = compiled._get_configure_command()
        LOGGER.debug('Creating configure subprocess in %r: %r' % (srcdir, cmd))
        proc = subprocess.Popen(cmd, cwd=srcdir, stdout=sys.stdout, stderr=sys.stderr)
        if proc.wait():
            LOGGER.critical('TAU configure failed.')
            raise TauError('TAU configure failed.')
        
        # Execute make
        cmd = ['make', '-j', 'install']
        LOGGER.debug('Creating make subprocess in %r: %r' % (srcdir, cmd))
        proc = subprocess.Popen(cmd, cwd=srcdir, stdout=sys.stdout, stderr=sys.stderr)
        if proc.wait():
            LOGGER.critical('TAU compilation failed.')
            raise TauError('TAU compilation failed.')
        
        # Mark this configuration as built
        self.built = True
        return compiled
    
    def getMakefile(self):
        makefiles = os.path.join(self.prefix, self.data['target'], 'lib', 'Makefile.tau*')
        return glob.glob(makefiles)[0]

    def getEnvironment(self):
        env = os.environ
        env['TAU_OPTIONS'] = ' '.join(TAU_OPTIONS)
        env['TAU_MAKEFILE'] = self.getMakefile()
        bindir = os.path.join(self.prefix, self.data['target'], 'bin')
        try:
            env['PATH'] = bindir + ':' + env['PATH']
            LOGGER.debug('Updated PATH to %r' % env['PATH'])
        except KeyError:
            LOGGER.warning('The PATH environment variable was unset.')
            env['PATH'] = bindir
        return env