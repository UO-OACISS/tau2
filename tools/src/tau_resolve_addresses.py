#!/usr/bin/env python
"""
@file
@author John C. Linford (jlinford@paratools.com)
@version 1.0

@brief 

Rewrites a profile file (possibly backing it up first) so that unresolved
addresses are resolved by addr2line.

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

import re
import sys
import shutil
from os import path
from subprocess import Popen, PIPE

class Addr2LineError(RuntimeError):
    def __init__(self, value):
        self.value = value


USAGE = "Usage: %(exe)s [-b|--backup] <file0> [<file1> ...]"

PATTERN = re.compile('UNRESOLVED (.*?) ADDR (0x[a-fA-F0-9]+)')


def die(msg, code=1):
    """
    Prints a message and exits
    """
    print msg
    sys.exit(code)
    

def addr2line(exe, addr):
    """
    Spawns an addr2line process to resolve an address
    """
    # Check that exe exists
    if not path.exists(exe):
        raise Addr2LineError('Profile references non-existent executable: %s' % exe)

    # Execute addr2line    
    cmd = ['addr2line', '-C', '-f', '-e', exe, addr]
    cmdstr = ' '.join(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, _ = proc.communicate()
    if proc.returncode != 0:
        raise Addr2LineError('addr2line failed: %s' % cmdstr)
    
    # Parse addr2line output
    parts = stdout.split()
    if len(parts) != 2:
        raise Addr2LineError('Unexpected output from %s: %s' (cmdstr, stdout))
    funcname = parts[0]
    if funcname == '??' or not len(funcname):
        funcname = '<%s>' % addr
    location_parts = parts[1].rsplit(':', 1)
    if len(location_parts) != 2:
        raise Addr2LineError('Unexpected output from %s: %s' (cmdstr, stdout))
    filename = location_parts[0]
    if filename == '??' or not len(filename):
        filename = exe
    lineno = location_parts[1]
    if lineno == '0':
        lineno = '-1'

    return funcname, filename, lineno


def resolve_addresses(line):
    """
    Uses addr2line to resolve addresses to function names and source code
    locations in a single line of profile data
    """
    def sub_repl(match):
        resolved = addr2line(match.group(1), match.group(2))
        return '%s [{%s} {%s}]' % (resolved[0], resolved[1], resolved[2])
    return re.sub(PATTERN, sub_repl, line)

def resolve_addresses_in_profile(fname):
    """
    Calls addr2line to resolve addresses in profile.* files
    """ 
    try:
        f = open(fname, 'r+b')
    except IOError:
        print 'Invalid filename: %s' % fname
        return
    
    # Count lines in profile and go back to file start
    num_lines = sum(1 for line in f)
    f.seek(0, 0) 
    
    # Read in file and rewrite in memory
    rewritten = list()
    for i, line in enumerate(f):
        if i % max((num_lines / 10), 1) == 0:
            perc = (float(i) / num_lines) * 100.0
            print 'Processing %s (%d%%)' % (fname, perc)
        try:
            rewritten.append(resolve_addresses(line))
        except Addr2LineError, e:
            print 'FATAL ERROR: ' % e.value
            f.close()
            return
    
    # Rewind the file and dump rewritten text
    f.seek(0, 0)
    f.writelines(rewritten)
    f.close()


def get_args():
    """
    Processes command line arguments from sys.argv
    """
    idx = 1
    backup = False
    
    if len(sys.argv) == 1:
        die(USAGE % {'exe': sys.argv[0]})
    
    if sys.argv[1] in ['-b', '--backup']:
        backup = True
        idx = idx + 1
        
    files = sys.argv[idx:]
    if not len(files):
        die(USAGE % {'exe': sys.argv[0]})
    
    return backup, files

if __name__ == '__main__':   
    backup, files = get_args()
    for fname in files:
        if backup:
            shutil.copy(fname, '%s.bak' % fname)
        resolve_addresses_in_profile(fname)
            
        
            