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
import pickle
import errno
import taucmd
from shutil import rmtree
from pprint import pformat
from hashlib import md5
from taucmd import TAU_ROOT_DIR


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
                parts.append('%s:\n%s' % (config['name'], pformat(config)))
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
    def load(cls, prefix, default):
        """
        Loads the configuration registry from file.
        """
        registry_file = os.path.join(prefix, 'registry')
        if os.path.exists(registry_file):
            with open(registry_file, 'rb') as f:
                registry = pickle.load(f)
                registry._registry_file = registry_file
                logging.info('Registry loaded from file %r' % registry_file)
                return registry
        else:
            logging.info('Registry file %r does not exist.' % registry_file)
            return cls(registry_file, prefix,  default)

    def save(self):
        """
        Saves the configuration registry to file.
        """
        try:
            os.makedirs(self.prefix)
            logging.info('Created directory %r' % self.prefix)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(self.prefix): pass
            else: raise
        with open(self._registry_file, 'wb') as f:
            pickle.dump(self, f)
            logging.info('Wrote registry file %r' % self._registry_file)

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
        for item in sorted(config.iteritems()):
            if item[0] != 'id':
                hash.update(repr(item))
        id = hash.hexdigest()
        config['id'] = id
        # Adjust prefix to include id
        config['prefix'] = os.path.join(self.prefix, id)
        # Register configuration
        self.data[name] = config
        # Set this configuration to default if it's the only configuration
        if len(self.data) == 1:
            self.set_default(name)
        # Create configuration prefix directory
        try:
            os.makedirs(config['prefix'])
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(config['prefix']): pass
            else: raise

    def unregister(self, name):
        """
        Removes a configuration from the registry
        """
        config = self.data[name]
        # Delete configuration files
        if os.path.isdir(config['prefix']):
            rmtree(config['prefix'], ignore_errors=True)
        # Remove from registry
        del self.data[name]
        # Change default if we just deleted the default configuration
        if name == self.default and len(self.data):
            self.set_default(self.data.values()[0]['name'])

    def set_default(self, name):
        """
        Records the name of the default configuration in the registry
        """
        if name in self.data:
            self.default = name
            logging.info('Changed default configuration to %r' % self.default)
        else:
            raise KeyError
    

def build(config):
    """
    Compiles a Tau configuration.
    """
    
    # Prepare the Tau code base
    print TAU_ROOT_DIR