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
import traceback
import textwrap

# Contact for bugs, etc.
HELP_CONTACT = '<tau-bugs@cs.uoregon.edu>'

# Logging level
LOG_LEVEL = 'INFO'

#Expected Python version
EXPECT_PYTHON_VERSION = (2, 7)

# Path to this package
PACKAGE_HOME = os.path.dirname(os.path.realpath(__file__))

# Search paths for included files
INCLUDE_PATH = [os.path.realpath('.')]

# User-specific TAU files
TAUCMD_HOME = os.path.join(os.path.expanduser('~'), '.tau')

# User-specific source code and build directory
SRC_DIR = os.path.join(TAUCMD_HOME, 'src')

# Tau source code root directory
try:
    TAU_MASTER_SRC_DIR = os.environ['TAU_MASTER_SRC_DIR']
except KeyError:
    print '!'*80
    print '!'
    print '! CRITICAL ERROR: TAU_MASTER_SRC_DIR environment variable not set.'
    print '!'
    print '!'*80
    exit(1)

DEFAULT_TAU_COMPILER_OPTIONS = ['-optRevert']


class TauError(Exception):
    """
    Base class for errors in Tau
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class TauConfigurationError(TauError):
    """
    Indicates that Tau cannot succeed with the given parameters
    """
    def __init__(self, value, hint="Try 'tau --help'."):
        super(TauConfigurationError,self).__init__(value)
        self.hint = hint

class TauNotImplementedError(TauError):
    """
    Indicates that a promised feature has not been implemented yet
    """
    def __init__(self, value, missing, hint="Try 'tau --help'."):
        super(TauNotImplementedError,self).__init__(value)
        self.missing = missing
        self.hint = hint

class TauUnknownCommandError(TauError):
    """
    Indicates that a specified command is unknown
    """
    def __init__(self, value, hint="Try 'tau --help'."):
        super(TauUnknownCommandError,self).__init__(value)
        self.hint = hint

class LogFormatter(logging.Formatter, object):
    """
    Custom log message formatter.
    """
    
    LINE_MARKER = 'TAU:'
    
    
    def __init__(self):
        super(LogFormatter, self).__init__()
        
    def debug_msg(self, record):
        return '%s %s:%s: %s' % (self.LINE_MARKER, record.levelname, record.module, record.getMessage()) 

    def info_msg(self, record):
        parts = []
        for line in record.getMessage().split('\n'):
            parts.append('%s %s' % (self.LINE_MARKER, line))
        return '\n'.join(parts)       
    
    def msgbox(self, record, marker):
        line_marker = self.LINE_MARKER
        hline = line_marker + marker*(79-len(line_marker))
        parts = [hline, line_marker, '%s %s' % (line_marker, record.levelname)]
        message = record.getMessage().split('\n')
        for line in message:
            parts.append('%s %s' % (line_marker, line))
        parts.append(line_marker)
        parts.append(hline)
        return '\n'.join(parts)

    def format(self, record):
        if record.levelno == logging.CRITICAL:
            return self.msgbox(record, '!')
        elif record.levelno == logging.ERROR:
            return self.msgbox(record, '!')
        elif record.levelno == logging.WARNING:
            return self.msgbox(record, '*')
        elif record.levelno == logging.INFO:
            return self.info_msg(record)
        elif record.levelno == logging.DEBUG:
            return self.debug_msg(record)
        else:
            return '%s:%s:%s' % (record.levelname, record.module, record.getMessage())

_loggers = set()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(LogFormatter())

def getLogger(name):
    """
    Returns a customized logging object by name
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(_handler)
    logger.propagate = False
    _loggers.add(logger)
    return logger

def setLogLevel(level):
    """
    Sets the output level for all logging objects
    """
    global LOG_LEVEL
    LOG_LEVEL = level.upper()
    _handler.setLevel(LOG_LEVEL)
    for logger in _loggers:
        logger.setLevel(LOG_LEVEL)

def excepthook(etype, e, tb):
    """
    Exception handler for any uncaught exception (except SystemExit).
    """
    logger = getLogger(__name__)
    if etype == KeyboardInterrupt:
        logger.info('Received keyboard interrupt.  Exiting.')
        sys.exit(1)
    elif etype == TauConfigurationError:
        hint = 'Hint: %s\n' % e.hint if e.hint else ''
        message = textwrap.dedent("""
        %(value)s
        %(hint)s
        Tau cannot proceed with the given inputs.
        Please review the input files and command line parameters
        or contact %(contact)s for assistance.""" % {'value': e.value, 'hint': hint, 'contact': HELP_CONTACT})
        logger.critical(message)
        sys.exit(-1)
    elif etype == TauNotImplementedError:
        hint = 'Hint: %s\n' % e.hint if e.hint else ''
        message = textwrap.dedent("""
        Unimplemented feature %(missing)r: %(value)s
        %(hint)s
        Sorry, you have requested a feature that is not yet implemented.
        Please contact %(contact)s for assistance.
        """ % {'missing': e.missing, 'value': e.value, 'hint': hint, 'contact': HELP_CONTACT})
        logger.critical(message)
        sys.exit(-1)
    elif etype == TauUnknownCommandError:
        hint = 'Hint: %s' % e.hint if e.hint else ''
        message = textwrap.dedent("""
        Unknown Command: %(value)r
        %(hint)s
        """ % {'value': e.value, 'hint': hint, 'contact': HELP_CONTACT})
        logger.info(message)
        sys.exit(-1)
    else:
        traceback.print_exception(etype, e, tb)
        args = [arg for arg in sys.argv[1:] if not '--log' in arg] 
        message = textwrap.dedent("""
        An unexpected %(typename)s exception was raised.
        Please contact <tau-bugs@cs.uoregon.edu> for assistance.
        If possible, please include the output of this command:

        tau --log=DEBUG %(cmd)s
        """ % {'typename': etype.__name__, 'cmd': ' '.join(args)})
        logger.critical(message)
        sys.exit(-1)
        
