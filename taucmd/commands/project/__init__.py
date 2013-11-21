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

import sys
import taucmd
from taucmd import commands
from taucmd.docopt import docopt

LOGGER = taucmd.getLogger(__name__)

SHORT_DESCRIPTION = "Create and manage TAU projects."

USAGE = """
Usage:
  tau project <command> [<args>...]
  tau project -h | --help

Project Commands:
%(command_descr)s
"""

HELP = """
'tau project' help page to be written.
"""

def getUsage():
    return USAGE % {'command_descr': commands.getSubcommands(__name__)}

def getHelp():
    return HELP

def main(argv):
    """
    Program entry point
    """

    # Parse command line arguments
    usage = getUsage()
    args = docopt(usage, argv=argv, options_first=True)
    LOGGER.debug('Arguments: %s' % args)
    
    # Check for -h | --help (why doesn't this work automatically?)
    idx = args['<command>'].find('-h')
    if (idx == 0) or (idx == 1):
        print usage
        return 0
    
    # Try to execute as a tau command
    cmd = args['<command>']
    cmd_args = args['<args>']
    cmd_module = 'taucmd.commands.project.%s' % cmd   
    
    try:
        __import__(cmd_module)
        LOGGER.debug('Recognized %r as tau project command' % cmd)
        return sys.modules[cmd_module].main(['project', cmd] + cmd_args)
    except ImportError:
        LOGGER.debug('%r not recognized as a tau project command' % cmd)
        print usage
        return 1
