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
import re
import subprocess
import errno
import taucmd
import urllib
import tarfile

LOGGER = taucmd.getLogger(__name__)


def mkdirp(path):
    """
    Creates a directory and all its parents.
    """
    try:
        os.makedirs(path)
        LOGGER.debug('Created directory %r' % path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path): pass
        else: raise

        
def which(program):
    def is_exec(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    fpath, _ = os.path.split(program)
    if fpath:
        if is_exec(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exec(exe_file):
                return exe_file
    return None


def download(src, dest):
    LOGGER.debug('Downloading %r to %r' % (src, dest))
    mkdirp(os.path.dirname(dest))
    curl = which('curl')
    LOGGER.debug('which curl: %r' % curl)
    wget = which('wget')
    LOGGER.debug('which wget: %r' % wget)
    if curl:
        if subprocess.call([curl, '-L', src, '-o', dest]) != 0:
            LOGGER.debug('curl failed to download %r.' % src)     
    elif wget:
        if subprocess.call([wget, src, '-O', dest]) != 0:
            LOGGER.debug('wget failed to download %r' % src)
    else:
        # Note, this is usually **much** slower than curl or wget
        def dlProgress(count, blockSize, totalSize):
            print "% 3.1f%% of %d bytes\r" % (min(100, float(count * blockSize) / totalSize * 100), totalSize),
        urllib.urlretrieve(src, dest, reporthook=dlProgress)
        
    
def extract(tgz, dest):
    with tarfile.open(tgz) as fp:
        LOGGER.debug('Determining top-level directory name in %r' % tgz)
        dirs = [d.name for d in fp.getmembers() if d.type == tarfile.DIRTYPE]
        topdir = min(dirs, key=len)
        LOGGER.debug('Top-level directory in %r is %r' % (tgz, topdir))
        full_dest = os.path.join(dest, topdir)
        LOGGER.debug('Extracting %r to create %r' % (tgz, full_dest))
        LOGGER.info('Extracting %r' % tgz)
        mkdirp(dest)
        fp.extractall(dest)
    assert os.path.isdir(full_dest)
    LOGGER.debug('Created %r' % full_dest)
    return full_dest

def getTauVersion():
    """
    Opens TAU header files to get the TAU version
    """
    header_files=['TAU.h', 'TAU.h.default']
    pattern = re.compile('#define\s+TAU_VERSION\s+"(.*)"')
    for hfile in header_files:
        try:
            with open('%s/include/%s' % (taucmd.TAU_MASTER_SRC_DIR, hfile), 'r') as tau_h:
                for line in tau_h:
                    match = pattern.match(line) 
                    if match:
                        return match.group(1)
        except IOError:
            continue
    return '(unknown)'