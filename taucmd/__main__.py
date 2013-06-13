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
import re
#import types
import signal
import textwrap
import taucmd
from docopt import docopt
from pkgutil import walk_packages
from taucmd import TauNotImplementedError
from taucmd import commands
from taucmd import compiler
from taucmd.registry import Registry 

USAGE = """
================================================================================
The Tau Performance System (version %(tau_version)s)
http://tau.uoregon.edu/

Usage:
  tau [options] <command> [<args>...]
  tau --version
  tau --help
  
  <command> may be a subcommand or compiler.  See tau --help.
  
Subcommands:
%(commands)s

Known Compilers:
%(compilers)s 

Options:
  --config=<name>  Tau configuration. %(config_default)s
  --home=<path>    Tau configuration home. [default: %(home_default)s]
  --log=<level>    Output level.  [default: %(log_default)s]
                     <level> can be CRITICAL, ERRROR, WARNING, INFO, or DEBUG
================================================================================
"""

LOGGER = taucmd.getLogger(__name__)

def lookupTauVersion():
    """
    Opens TAU.h to get the TAU version
    """
    if not taucmd.TAU_ROOT_DIR:
        return '(unknown)'
    with open('%s/include/TAU.h' % taucmd.TAU_ROOT_DIR, 'r') as tau_h:
        pattern = re.compile('#define\s+TAU_VERSION\s+"(.*)"')
        for line in tau_h:
            match = pattern.match(line) 
            if match:
                return match.group(1)
    return '(unknown)'

def lookupDefaultConfig():
    """
    Loads the registry to get the default config.
    """
    registry = Registry.load()
    if not len(registry):
        return ''
    else:
        return '[default: %s]' % registry.default

def getKnownCompilers():
    """
    Returns a string listing known compiler commands
    """
    known = ', '.join(compiler.knownCompilerCommands())
    return textwrap.fill(known, width=70, initial_indent='  ', subsequent_indent='  ')

def getCommandList():
    """
    Builds listing of command names with short description
    """
    parts = []
    for module in [name for _, name, _ in walk_packages(path=commands.__path__, prefix=commands.__name__+'.')]:
        __import__(module)
        descr = sys.modules[module].SHORT_DESCRIPTION
        name = '{:<15}'.format(module.split('.')[-1])
        parts.append('  %s  %s' % (name, descr))
    return '\n'.join(parts)


        
def main():
    """
    Program entry point
    """

    # Set the default exception handler
    sys.excepthook = taucmd.excepthook

    # Check Python version
    if sys.version_info < taucmd.EXPECT_PYTHON_VERSION:
        version = '.'.join(map(str, sys.version_info[0:3]))
        expected = '.'.join(map(str, taucmd.EXPECT_PYTHON_VERSION))
        LOGGER.warning("Your Python version is %s, but 'tau' expects Python %s or later.  Please update Python." % (version, expected))

    # Get tau version
    tau_version = lookupTauVersion()
    
    # Parse command line arguments
    usage = USAGE % {'tau_version': tau_version,
                     'home_default': taucmd.HOME,
                     'config_default': lookupDefaultConfig(),
                     'log_default': taucmd.LOG_LEVEL,
                     'compilers': getKnownCompilers(),
                     'commands': getCommandList()}
    args = docopt(usage, version=tau_version, options_first=True)

    # Set log level
    taucmd.setLogLevel(args['--log'])
    LOGGER.debug('Arguments: %s' % args)
    LOGGER.debug('Verbosity level: %s' % taucmd.LOG_LEVEL)
    
    # Record global arguments
    taucmd.HOME = args['--home']
    taucmd.CONFIG = args['--config']
    
    # Try to execute as a tau command
    cmd = args['<command>']
    cmd_args = args['<args>']
    cmd_module = 'taucmd.commands.%s' % cmd
    try:
        __import__(cmd_module)
        LOGGER.debug('Recognized %r as tau subcommand' % cmd)
        return sys.modules[cmd_module].main([cmd] + cmd_args)
    except ImportError:
        # It wasn't a tau command, but that's OK
        LOGGER.debug('%r not recognized as tau subcommand' % cmd)

    # Try to execute as a compiler command
    try:
        retval = compiler.compile(args)
        if retval == 0:
            LOGGER.info('Compilation successful')
        elif retval > 0:
            LOGGER.critical('Compilation failed')
        elif proc.returncode < 0:
            signal_names = dict((getattr(signal, n), n) for n in dir(signal) 
                                if n.startswith('SIG') and '_' not in n)
            LOGGER.critical('Compilation aborted by signal %s' % signal_names[-proc.returncode])
        return retval
    except TauNotImplementedError:
        # It wasn't a compiler command, but that's OK
        LOGGER.debug('%r not recognized as a compiler command' % cmd)
        
    # Not sure what to do at this point, so advise the user and exit
    LOGGER.debug("Can't classify %r.  Calling 'tau help' to get advice." % cmd)
    return commands.help.main(['help', cmd])


# Command line execution
if __name__ == "__main__":
    exit(main())