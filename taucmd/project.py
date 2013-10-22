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
import taucmd
import pickle
import pprint
from datetime import datetime
from taucmd.installers import pdt, bfd, tau
from taucmd import util


LOGGER = taucmd.getLogger(__name__)

    
class ProjectNameError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Registry(object):
    """
    TODO: Docs
    """
    def __init__(self, projects_dir='.tau'):
        self.projects_dir = projects_dir
        self.projects = None
        self.load()
        
    def __str__(self):
        return pprint.pformat(self.projects)

    def load(self):
        if self.projects:
            LOGGER.debug('Project registry already loaded.')
        try:
            file_path = os.path.join(self.projects_dir, 'projects')
            with open(file_path, 'rb') as fp:
                self.projects = pickle.load(fp)
            LOGGER.debug('Project registry loaded from %r' % file_path)
        except:
            LOGGER.debug('Project registry file %r does not exist.' % file_path)
            self.projects = {}

    def save(self):
        util.mkdirp(self.projects_dir)
        file_path = os.path.join(self.projects_dir, 'projects')
        with open(file_path, 'wb') as fp:
            pickle.dump(self.projects, fp)
        LOGGER.debug('Project registry written to %r' % file_path)

    def getDefaultProject(self):
        projects = self.projects
        if projects:
            default = projects['default']
            return Project(self, projects[default])
        return None
    
    def setDefaultProject(self, proj_name):
        # Set a project as default
        projects = self.projects
        if not proj_name in projects:
            raise ProjectNameError("Error: No project named %r exists.  See 'tau project list' for project names." % proj_name)
        # Update default
        projects['default'] = proj_name
        self.save()

    def newProject(self, args):
        # Strip and check args
        config = {'refresh': True,
                  'modified': datetime.now()}
        exclude = ['--help', '-h', '--default']
        for key, val in args.iteritems():
            if key[0:2] == '--' and not key in exclude:
                if key == '--name' and val == 'default':
                    raise ProjectNameError("Error: 'default' cannot be used as a project name.  Use 'tau project select' to set the default project.")
                elif key == '--pdt':
                    if val.upper() == 'NONE':
                        config['pdt'] = None
                    elif val.upper() == 'DOWNLOAD':
                        config['pdt'] = 'download'
                    else:
                        config['pdt'] = val
                elif key == '--bfd':
                    if val.upper() == 'NONE':
                        config['bfd'] = None
                    elif val.upper() == 'DOWNLOAD':
                        config['bfd'] = 'download'
                    else:
                        config['bfd'] = val
                else:
                    config[key[2:]] = val
        # Create the project object and update the registry
        proj = Project(self, config)
        proj_name = proj.getName()
        projects = self.projects
        if proj_name in projects:
            raise ProjectNameError("Error: Project %r already exists.  See 'tau project create --help' and maybe use the --name option." % proj_name)
        projects[proj_name] = proj.config
        if args['--select'] or not 'default' in projects:
            projects['default'] = proj_name
        self.save()
        return proj

    def deleteProject(self, proj_name):
        projects = self.projects
        try:
            del projects[proj_name]
            LOGGER.debug('Removed %r from project registry' % proj_name)
        except KeyError:
            raise ProjectNameError('Error: No project named %r.' % proj_name)
        # Update default if necessary
        new_default = None
        if projects['default'] == proj_name:
            if len(projects) > 1:
                for new_default in projects.iterkeys():
                    if new_default != 'default':
                        break
                projects['default'] = new_default
            else:
                self.projects = {}
        # Save registry
        self.save()
        # TODO: Delete project files
        


class Project(object):
    """
    TODO: DOCS
    """
    def __init__(self, registry, config):
        config['tau-prefix'] = tau.getPrefix(config)
        config['pdt-prefix'] = pdt.getPrefix(config)
        config['bfd-prefix'] = bfd.getPrefix(config)
        self.registry = registry
        self.config = config

    def getName(self):
        config = self.config
        if config['name']:
            return config['name']
        else:
            nameparts = ['bfd', 'binary-inst', 'callpath', 'comm-matrix', 'compiler-inst', 
                         'cuda', 'dyninst', 'io', 'memory', 'memory-debug', 'mpi', 'openmp',
                         'papi', 'pdt', 'profile', 'pthreads', 'sample', 'source-inst', 'trace']
            valueparts = ['c++', 'cc', 'fortran', 'target-arch', 'upc', 'upc-network']
            parts = [config[part].lower() for part in valueparts if config[part]]
            parts.extend([part.lower() for part in nameparts if config[part]])
            parts.sort()
            name = '_'.join(parts)
            config['name'] = name
            return name

    def getCompilers(self):
        compiler_fields = ['cc', 'c++', 'fortran', 'upc']
        return {key: self.config[key] for key in compiler_fields}
    
    def supportsCompiler(self, cmd):
        return cmd in self.getCompilers().values()
    
    def supportsExec(self, cmd):
        config = self.config
        if cmd in ['mpirun', 'mpiexec']:
            return bool(config['mpi'])
        return True

    def compile(self):
        config = self.config
        if not config['refresh']:
            return
        
        print '*' * 80
        print '*'
        print '* Compiling project %r.' % config['name']
        print '* This may take a long time but will only be done once.'
        print '*'
        print '*' * 80

        # Control configure/build output
        devnull = None
        if taucmd.LOG_LEVEL == 'DEBUG':
            stdout = sys.stdout
            stderr = sys.stderr
        else:
            devnull = open(os.devnull, 'w')
            stdout = devnull
            stderr = devnull
        
        # Build PDT, BFD, TAU as needed
        pdt.install(config, stdout, stderr)
        bfd.install(config, stdout, stderr)
        tau.install(config, stdout, stderr)

        # Mark this configuration as built
        if devnull:
            devnull.close() 
        config['refresh'] = False
        config['modified'] = datetime.now()
        self.registry.save()

    def getTauMakefile(self):
        """
        Returns TAU_MAKEFILE for this configuration
        """
        config = self.config
        makefiles = os.path.join(config['tau-prefix'], config['target-arch'], 'lib', 'Makefile.tau*')
        makefile = glob.glob(makefiles)[0]
        LOGGER.debug('TAU Makefile: %r' % makefile)
        return makefile
    
    def getTauTags(self):
        """
        Returns TAU tags for this project
        """
        makefile = self.getTauMakefile()
        start = makefile.find('Makefile.tau')
        tags = makefile[start+12:].split('-')
        if len(tags) > 1:
            return map(lambda x: x.upper(), tags[1:])
        return []
        
    def getTauCompilerEnvironment(self):
        """
        Returns an environment for use with subprocess.Popen that specifies the
        compile-time TAU environment variables for this project
        """
        config = self.config
        env = dict(os.environ)
        env['TAU_OPTIONS'] = ' '.join(taucmd.DEFAULT_TAU_COMPILER_OPTIONS)
        env['TAU_MAKEFILE'] = self.getTauMakefile()
        bindir = os.path.join(config['tau-prefix'], config['target-arch'], 'bin')
        try:
            env['PATH'] = bindir + ':' + env['PATH']
            LOGGER.debug('Updated PATH to %r' % env['PATH'])
        except KeyError:
            LOGGER.warning('The PATH environment variable was unset.')
            env['PATH'] = bindir
        return env

    def getTauCompilerFlags(self):
        """
        Returns compiler flags for the TAU compiler wrappers (tau_cc.sh, etc.)
        """
        # Perhaps someday...
        return []
    
    def getTauExecEnvironment(self):
        """
        Returns an environment for use with subprocess.Popen that specifies the
        run-time TAU environment variables for this project
        """
        config = self.config
        flags = {'callpath': ['TAU_CALLPATH'],
                  'comm-matrix': ['TAU_COMM_MATRIX'],
                  'memory': ['TAU_TRACK_HEAP', 'TAU_TRACK_MEMORY_LEAKS'],
                  'memory-debug': ['TAU_MEMDBG_PROTECT_ABOVE', 'TAU_TRACK_MEMORY_LEAKS'],
                  'profile': ['TAU_PROFILE'],
                  'sample': ['TAU_SAMPLING'],
                  'trace': ['TAU_TRACE']}
        env = dict(os.environ)
        for key, val in config.iteritems():
            if key in flags and val:
                env.update([(x, 1) for x in flags[key]])
        if config['callpath']:
            env['TAU_CALLPATH_DEPTH'] = config['callpath-depth']
        return env

    def getTauExecFlags(self):
        """
        Returns tau_exec flags
        """
        tags = self.getTauTags()
        print tags
        