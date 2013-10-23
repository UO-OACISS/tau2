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
import re
import subprocess
import taucmd
from threading import Thread
from taucmd import util
from taucmd.docopt import docopt
from taucmd.project import Registry

LOGGER = taucmd.getLogger(__name__)

SHORT_DESCRIPTION = "Display application profile or trace data."

USAGE = """
Usage:
  tau show [<files>...]
  tau show -h | --help
  
<files> may be profile files (profile.*, *.ppk, *.xml, etc.) or traces (*.otf, *.slog2).
If not files are given, show all files in current directory.
"""

HELP = """
'tau show' help page to be written.
"""


FILE_VIEWERS = {'profile': 'paraprof',
                'ppk': 'paraprof',
                'xml': 'paraprof',
                'otf': 'jumpshot',
                'slog2': 'jumpshot'}

PROFILE_PATTERN = re.compile('^profile\.-?\d+\.\d+\.\d+$')


def getUsage():
    return USAGE

def getHelp():
    return HELP

def isProfileFile(filename):
    return PROFILE_PATTERN.match(filename) != None

def isKnownFileFormat(fname):
    return isProfileFile(fname) or fname.split('.')[-1] in FILE_VIEWERS 

def launchViewer(cmd, env):
    LOGGER.debug('Creating subprocess: cmd=%r' % cmd)
    subprocess.call(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)

def main(argv):
    """
    Program entry point
    """
    # Parse command line arguments
    usage = getUsage()
    args = docopt(usage, argv=argv)
    LOGGER.debug('Arguments: %s' % args)
    
    # Check project compatibility
    registry = Registry()
    proj = registry.getSelectedProject()
    if not proj:
        LOGGER.info("There are no TAU projects in %r.  See 'tau project create'." % os.getcwd())
        return 1
    LOGGER.info('Using TAU project %r' % proj.getName())
    
    args_files = args['<files>']
    if not args_files:
        args_files = glob.glob('profile.*.*.*')
        for ext in FILE_VIEWERS:
            if ext != 'profile':
                args_files += glob.glob('*.%s' % ext)
        LOGGER.debug('Found files: %r' % args_files)
        
    # Compile the project if needed
    proj.compile()
    
    # Sort files by type
    all_files = {}
    for arg in args_files:
        if isProfileFile(arg):
            if 'profile' not in all_files:
                all_files['profile'] = []
            all_files['profile'].append(arg)
        else:
            ext = arg.split('.')[-1]
            if ext not in all_files:
                all_files[ext] = []
            all_files[ext].append(arg)
    LOGGER.debug('Sorted files: %r' % all_files)
    
    # Construct commands
    cmds = []
    for filetype, files in all_files.iteritems():
        try:
            viewer = FILE_VIEWERS[filetype]
        except KeyError:
            LOGGER.error('Unknown file type %r' % filetype)
            continue 
        cmd = [viewer] + files
        cmds.append(cmd)
    LOGGER.debug('Commands: %r' % cmds)
    
    # Get environment
    env = proj.getEnvironment()
    
    # Launch viewers
    threads = [Thread(target=launchViewer, args=(cmd,env)) for cmd in cmds]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return 0