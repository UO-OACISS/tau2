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
import time
from glob import glob
from mmap import mmap
from optparse import OptionParser
from subprocess import Popen, PIPE
from threading import Thread
from Queue import Queue, Empty
from xml.sax import saxutils

USAGE = """
%prog [options] tauprofile.xml
"""

PATTERN = re.compile('(.*?)UNRESOLVED (.*?) ADDR (0x[a-fA-F0-9]+)')

# Seconds
TIMEOUT = 300

# How many times do the workers report?
ITERS_PER_REPORT = 5000

isXML = False

class Addr2LineError(RuntimeError):
    def __init__(self, value):
        self.value = value


class Addr2Line(object):

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

    def __init__(self, addr2line, exe):
        cmd = [addr2line, '-C', '-f', '-e', exe]
        self.exe = exe
        self.cmdstr = ' '.join(cmd)
        if not os.path.exists(self.exe):
            print 'WARNING: %r not found.  Addresses in this binary will not be resolved.' % self.exe
            self.p = self.q = self.t = None
        else:
            self.p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1)
            self.q = Queue()
            self.t = Thread(target=Addr2Line.enqueue_output, args=(self.p.stdout, self.q))
            self.t.daemon = True
            self.t.start()
            print 'New process: %s' % self.cmdstr

    def close(self):
        if self.p:
            self.q.join()
            self.p.stdin.close()
            retval = self.p.wait()
            if retval != 0:
                raise Addr2LineError('Nonzero exit code %d from %r\nstdout: %s\nstderr: %s' % 
                                     (retval, self.cmdstr, self.p.stdout, self.p.stderr))

    def resolve(self, addr):
        # Send address to addr2line
        if self.p:
            self.p.stdin.write(addr+'\n')
    
            # Read addr2line output 
            try:
                funcname, location = self.q.get(block=True,timeout=TIMEOUT)
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
                filename = self.exe
            return funcname, filename, lineno
        else:
            return 'UNRESOLVED', self.exe, '0'


class Worker(Thread):

    def __init__(self, addr2line, all_exes, mm, unresolved, linespan, chunk):
        Thread.__init__(self)
        self.results = list()
        self.mm = mm
        self.unresolved = unresolved
        self.linespan = linespan
        self.firstline = chunk[0]
        self.lastline = self.firstline + chunk[1]
        self.linecount = chunk[1]
        self.pipes = dict()
        for exe in all_exes:
            self.pipes[exe] = Addr2Line(addr2line, exe)

    def run(self):
        global isXML
        """
        """
        def repl(match):
            prefix = match.group(1)
            exe = match.group(2)
            addr = match.group(3)
            if exe == 'UNKNOWN':
                for p in self.pipes.itervalues():
                    resolved = p.resolve(addr)
                    if resolved[0] != 'UNRESOLVED':
                        break
            else:
                resolved = self.pipes[exe].resolve(addr)
            if resolved[0] != 'UNRESOLVED':
                if isXML:
                    return '%s%s [{%s} {%s}]' % (prefix, saxutils.escape(resolved[0]), saxutils.escape(resolved[1]), saxutils.escape(resolved[2]))
                else:
                    return '%s%s [{%s} {%s}]' % (prefix, resolved[0], resolved[1], resolved[2])
            else:
                return match.group(0)

        # Extract lines from memory and rewrite
        print 'New thread: %s' % self.name
        t0 = time.clock()
        j = 1
        for i in xrange(self.firstline, self.lastline):
            if j % ITERS_PER_REPORT == 0:
                timespan = time.clock() - t0
                time_per_iter = timespan / ITERS_PER_REPORT
                eta = (self.linecount - j) * time_per_iter
                etadate = time.ctime(time.time() + eta)
                print '%s: %d records in %f seconds, ETA %f seconds (%s)' % (self.name, ITERS_PER_REPORT, timespan, eta, etadate)
                if eta > 1000:
                    print 'This is going to take a long time. Maybe use the --jobs option? See --help'
                t0 = time.clock()
            lineno = self.unresolved[i]
            span = self.linespan[lineno]
            start = span[0]
            stop = span[1]
            line = self.mm[start:stop]
            try:
                result = (lineno, re.sub(PATTERN, repl, line))
            except Addr2LineError, e:
                print e.value
                break
            self.results.append(result)
            j += 1


def tauprofile_xml(infile, outfile, options):
    global isXML
    """
    Calls addr2line to resolve addresses in a tauprofile.xml file
    """ 
    fallback_exes = set(options.exe)
    addr2line = options.addr2line
    jobs = int(options.jobs)
    
    if "xml" in infile:
        isXML = True

    with open(infile, 'r+b') as fin:
        with open(outfile, 'wb') as fout:

            # Scan events from input file
            print 'Scanning %r' % infile
            all_exes = set()
            linespan = list()
            unresolved = list()
            offset = 0
            t0 = time.clock()
            j = 1
            for line in fin:
                # There are no event records after the first </definitions> tag
                if line.startswith('</definitions>'):
                    break
                if j % ITERS_PER_REPORT == 0:
                    timespan = time.clock() - t0
                    print 'Scanned %d lines in %f seconds' % (j, timespan)
                linespan.append((offset, offset + len(line)))
                offset += len(line)
                match = re.search(PATTERN, line)
                if match:
                    unresolved.append(len(linespan) - 1)
                    exe = match.group(2)
                    if exe != 'UNKNOWN':
                        all_exes.add(exe)
                j += 1
            linecount = len(unresolved)

            # "Rewind" the input file and report
            fin.seek(0, 0)
            print 'Found %d executables in profile' % len(all_exes)
            print 'Found %d unresolved addresses' % linecount
            if jobs > linecount:
                jobs = linecount
                print 'Reducing jobs to %d' % jobs

            # Build list of executables to search
            all_exes |= fallback_exes
            if not all_exes:
                print 'ERROR: No executables or other binary objects specified. See --help.'
                sys.exit(1)

            # Calculate work division
            chunks = list()
            start = 0
            if jobs > 1:
                chunklen = linecount / jobs
                chunkrem = linecount % jobs
                for i in xrange(jobs):
                    count = chunklen
                    if i < chunkrem:
                        count += 1
                    chunks.append((start, count))
                    start += count
                print '%d workers process %d records, %d process %d records' % (chunkrem, chunklen+1, (jobs-chunkrem), chunklen)
            else:
                chunks = [(0, linecount)]
                print 'One thread will process %d records' % linecount

            # Launch worker processes
            mm = mmap(fin.fileno(), 0)
            workers = list()
            for i in xrange(jobs):
                w = Worker(addr2line, all_exes, mm, unresolved, linespan, chunks[i])
                w.start()
                workers.append(w)

            # if there were no unresolved symbols, just copy the profile
            if len(workers) == 0:
                fin.seek(0, 0)
                for line in fin:
                    fout.write(line)

            i = 0
            # Process worker output
            for rank, w in enumerate(workers):
                w.join()
                print '%s (%d/%d) completed' % (w.name, rank, len(workers))
                for lineno, line in w.results:
                    if i < lineno:
                        start = linespan[i][0]
                        stop = linespan[lineno-1][1]
                        print 'writing lines %d:%d' % (i, lineno-1)
                        fout.write(mm[start:stop])
                        i = lineno
                    fout.write(line)
                    i += 1

            # Write out remainder of file
            print 'Address resolution complete, writing metrics to file...'
            start = linespan[i-1][1]
            fin.seek(start, 0)
            for line in fin:
                fout.write(line)


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
    parser.add_option('-j', '--jobs', help='Number of parallel jobs to use', default=1)
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

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for infile in files:
        outfile = os.path.join(outdir, os.path.basename(infile))
        print '%s => %s' % (infile, outfile)
        try:
            tauprofile_xml(infile, outfile, options)
        except IOError:
            print 'Invalid input or output file.  Check command arguments'
            sys.exit(1)

