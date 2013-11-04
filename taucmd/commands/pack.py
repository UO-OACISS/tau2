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
import taucmd
from taucmd import util
from taucmd.docopt import docopt
from taucmd.project import Registry

LOGGER = taucmd.getLogger(__name__)

SHORT_DESCRIPTION = "Package profile files into a PPK file."

USAGE = """
Usage:
  tau pack [options] [<profile>...]
  tau pack -h | --help
  
Options:
  --name=<name>            Specify the PPK file name.
  --rm-profiles            Delete profile.* files after creating PPK file.
  --no-project-name        Do not include the project name in the PPK file name.
"""

HELP = """
'tau pack' help page to be written.
"""

def getUsage():
    return USAGE

def getHelp():
    return HELP

def isExecutable(cmd):
    return util.which(cmd) != None

def main(argv):
    """
    Program entry point
    """
    # Parse command line arguments
    args = docopt(USAGE, argv=argv)
    LOGGER.debug('Arguments: %s' % args)

    # Get selected project
    registry = Registry()
    proj = registry.getSelectedProject()
    if not proj:
        LOGGER.info("There are no TAU projects in %r.  See 'tau project create'." % os.getcwd())
        return 1

    # Check for profiles
    profiles = args['<profile>']
    if not profiles:
        profiles = glob.glob('profile.*.*.*')
        if not profiles:
            LOGGER.error('No profile files in %r' % os.getcwd())
            return 1

    # Get project name
    if args['--no-project-name']:
        proj_name = ''
    else:
        proj_name = proj.getName()
        LOGGER.info('Using TAU project %r' % proj.getName())
    
    # Get PPK file name
    name = args['--name']
    if not name:
        name = 'tau'
    
    # Pack the profiles
    if proj_name:
        ppk_name = '%s.%s.ppk' % (name, proj_name)
    else:
        ppk_name = '%s.ppk' % name
    cmd = ['paraprof', '--pack', ppk_name]
    proc = subprocess.Popen(cmd, env=proj.getEnvironment(), stdout=sys.stdout, stderr=sys.stderr)
    retval = proc.wait()
    if retval < 0:
        LOGGER.error('paraprof killed by signal %d' % -retval)
    elif retval > 0:
        LOGGER.error('paraprof failed with exit code %d' % retval)
    elif args['--rm-profiles']:
        LOGGER.debug('Removing profiles: %r' % profiles)
        for profile in profiles:
            os.remove(profile)
    return retval