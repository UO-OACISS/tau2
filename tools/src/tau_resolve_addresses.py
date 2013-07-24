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

from __future__ import with_statement

import re
import os
import sys
from glob import glob
from optparse import OptionParser
from subprocess import Popen, PIPE
from threading import Thread
from Queue import Queue, Empty

USAGE = """
%prog [options] [file1 file2 ...]

Input files may be regular profiles (profile.0.0.0, profile.1.0.0, etc.) or 
merged profiles (tauprofile.xml).  If no input files are given, files named
profile.* in the current directory are used as input. See -h for details."""

PATTERN = re.compile('UNRESOLVED (.*?) ADDR (0x[a-fA-F0-9]+)')


class Addr2LineError(RuntimeError):
    def __init__(self, value):
        self.value = value


class Addr2Line(object):

    _instances = dict()

    @classmethod
    def enqueue_output(cls, out, queue):
        flag = False
        last = None
        for line in iter(out.readline, b''):
            line = line.strip()
            if flag:
                queue.put((last, line))
                flag = False
            else:
                last = line
                flag = True
        out.close()

    @classmethod
    def resolve(cls, exe, addr):
        if exe == 'UNKNOWN':
            for addr2line in Addr2Line._instances.itervalues():
                resolved = addr2line._resolve(addr)
                if resolved[0] != 'UNRESOLVED':
                    break
            return resolved
        else:
            return Addr2Line._instances[exe].resolve(addr)

    def __init__(self, addr2line, exe):
        Addr2Line._instances[exe] = self
        cmd = [addr2line, '-C', '-f', '-e', exe]
        self.exe = exe
        self.cmdstr = ' '.join(cmd)
        self.p = Popen(cmd, stdin=PIPE, stdout=PIPE, bufsize=1)
        self.q = Queue()
        self.t = Thread(target=Addr2Line.enqueue_output, args=(self.p.stdout, self.q))
        self.t.daemon = True
        self.t.start()

    def close(self):
        self.p.stdin.close()
        retval = self.p.wait()
        if retval != 0:
            raise Addr2LineError('Nonzero exit code %d from %r' % (retval, self.cmdstr))
    
    def _resolve(self, addr):
        # Send address to addr2line
        self.p.stdin.write(addr+'\n')

        # Read addr2line output 
        try:
            funcname, location = self.q.get(block=True,timeout=120)
        except Empty:
            raise Addr2LineError('ERROR: %r timed out resolving address %r' % (self.cmdstr, addr))

        # Parse location
        location_parts = location.rsplit(':', 1)
        if len(location_parts) != 2:
            raise Addr2LineError('Unexpected output from %r: %r' % (cmdstr, location))
        filename = location_parts[0]
        lineno = location_parts[1]

        # Return results
        if not funcname or funcname == '??':
            funcname = 'UNRESOLVED'
        if not filename or filename == '??':
            filename = 'UNKNOWN'
        return funcname, filename, lineno


def resolve(match):
    """
    """
    exe = match.group(1)
    addr = match.group(2)
    resolved = Addr2Line.resolve(exe, addr)
    return '%s [{%s} {%s}]' % (resolved[0], resolved[1], resolved[2])


def resolve_addresses_in_profile(infile, outfile, addr2line, fallback_exes):
    """
    Calls addr2line to resolve addresses in profile.* files
    """ 
    with open(infile, 'r+') as fin:
        with open(outfile, 'w') as fout:
            # Scan input file for executable names
            print 'Scanning %r' % infile
            all_exes = list()
            linecount = 0
            for line in fin:
                linecount = linecount + 1
                match = re.search(PATTERN, line)
                if match:
                    exe = match.group(1)
                    if exe != 'UNKNOWN':
                        all_exes.append(exe)

            # Build list of executables to search
            all_exes.extend(fallback_exes)
            if not all_exes:
                print 'ERROR: No executables or other binary objects specified. See --help.'
                return

            # Spawn addr2line threads
            print 'Spawning addr2line'
            for exe in all_exes:
                Addr2Line(addr2line, exe)

            # Resolve addresses
            fin.seek(0, 0)
            for i, line in enumerate(fin):
                perc = int((float(i) / linecount) * 100.0)
                if i % 1000 == 0:
                    print '%d of %d records processed (%d%%)' % (i, linecount, perc)
                fout.write(re.sub(PATTERN, resolve, line))


def which(program):
    """
    Returns full path to an executable or None if the named
    executable is not in the path.
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def get_args():
    """
    Parse command line arguments
    """
    parser = OptionParser(usage=USAGE)
    parser.add_option('-o', '--outdir', help='Specify output directory.', default='RESOLVED')
    parser.add_option('-e', '--exe', help='Add binary to list of files to search. Repeatable.', 
                      action='append', default=[])
    parser.add_option('-a', '--addr2line', help='Command to execute as addr2line.', 
                      default='addr2line', metavar='CMD')
    (options, args) = parser.parse_args()

    # Check executables
    for exe in options.exe:
        if not os.path.exists(exe):
            parser.error('Invalid binary: %r' % exe)

    # Check addr2line command
    if not which(options.addr2line):
        parser.error('addr2line command not found in PATH: %r' % options.addr2line)
    
    # Check input files
    files = args if args else glob('profile.*')
    if not files:
        parser.error('At least one profile file must be specified.')

    return options, files


if __name__ == '__main__':   
    options, files = get_args()
    outdir = options.outdir
    exes = options.exe
    addr2line = options.addr2line

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for infile in files:
        outfile = os.path.join(outdir, infile)
        try:
            resolve_addresses_in_profile(infile, outfile, addr2line, exes)
        except IOError:
            print 'Invalid input or output file.  Check command arguments'
            sys.exit(1)

