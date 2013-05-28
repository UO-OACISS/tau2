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
import logging
from docopt import docopt

USAGE = """
Usage:
  tau help <command>


"""

SHORT_DESCRIPTION = "Get help with a command."

HELP = """
Prints the help page for a specified command.
"""

MAKE_ADVICE = """
'make' is not a Tau command.

--- Did you try to build your codebase by typing 'tau make'?  

Tau cannot rewrite your makefiles for you, but it's fairly easy to do yourself.
All you need to do is put the 'tau' command before your compiler invocation.  
So, if your makefile contains lines something like this:
    CC = gcc
you'll change it to:
    CC = tau gcc
Be sure to do this for all instances of CC, CXX, F90, etc.
Sometimes you can override CC, CXX, etc. from the command line like this:
> make CC="tau gcc" CXX="tau g++" F90="tau gfortran"
Don't forget the double quotes!

--- Did you want to gather performance information on the 'make' command?

If you typed 'tau make' to get performance data on the make command, you should
choose the kind of data you want to gather and use the appropriate subcommand.
For example, to gather profiling data type:
    tau profile make <args>

Type 'tau --help' to see a complete list of subcommands.
"""

def advise(cmd):
    """
    Print some advice about a system command.
    """
    
    if cmd == 'make':
        print MAKE_ADVICE
        return 0 
    else:
        print "%r: Unknown command. Try 'tau --help'." % cmd
        return 1

def main(argv):
    """
    Program entry point
    """
    
    # Parse command line arguments
    args = docopt(USAGE, argv=argv)
    logging.debug('Arguments: %s' % args)
    
    # Try to look up a Tau command's built-in help page
    cmd = args['<command>']
    cmd_module = 'taucmd.commands.%s' % cmd
    try:
        __import__(cmd_module)
        logging.info('Recognized %r as tau subcommand' % cmd)
        print sys.modules[cmd_module].HELP
        return 0
    except ImportError:
        # It wasn't a tau command, but that's OK
        logging.debug('%r not recognized as tau subcommand' % cmd)

    # Do our best to give advice about this strange command
    return advise(cmd)


if __name__ == '__main__':
    exit(main(['help'] + sys.argv))