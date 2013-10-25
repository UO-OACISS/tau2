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
import subprocess
import taucmd
import shutil
from taucmd import util
from taucmd import TauError, TauNotImplementedError

LOGGER = taucmd.getLogger(__name__)

# Default PDT download URL
PDT_URL = 'http://tau.uoregon.edu/pdt.tgz'
# For debugging
#PDT_URL = 'http://localhost:3000/pdtoolkit-3.19p1.tar.gz'
#PDT_URL = 'http://www.cs.uoregon.edu/research/tau/pdt_releases/pdtoolkit-3.19p1.tar.gz'

# User specific PDT source code location
PDT_SRC_DIR = os.path.join(taucmd.SRC_DIR, 'pdt')



def downloadSource(src=PDT_URL):
    """
    Downloads and extracts a PDT archive file
    """
    dest = os.path.join(taucmd.SRC_DIR, src.split('/')[-1])
    LOGGER.info('Downloading PDT from %r' % src)
    util.download(src, dest)
    extractSource(dest)
    os.remove(dest)


def extractSource(tgz):
    """
    Extracts a PDT archive file
    """
    pdt_path = util.extract(tgz, taucmd.SRC_DIR)
    shutil.rmtree(PDT_SRC_DIR, ignore_errors=True)
    shutil.move(pdt_path, PDT_SRC_DIR)


def getConfigureCommand(config):
    """
    Returns the command that will configure PDT for a project
    """
    # TODO: Support other compilers
    return ['./configure', '-GNU', '-prefix=%s' % config['pdt-prefix']]


def getPrefix(config):
    # TODO: Support other compilers
    prefix = os.path.join(taucmd.TAUCMD_HOME, 'pdt', 'GNU')
    return prefix


def install(config, stdout=sys.stdout, stderr=sys.stderr):
    """
    Installs PDT
    """
    pdt = config['pdt']
    prefix = getPrefix(config)
    if not pdt or os.path.isdir(prefix):
        LOGGER.debug("Skipping PDT installation: pdt=%r, prefix=%r" % (pdt, prefix))
        return

    # Download and extract PDT if needed
    if pdt == 'download':
        downloadSource()
    elif os.path.isfile(pdt):
        extractSource(pdt)
    elif os.path.isdir(pdt):
        LOGGER.debug('Assuming user-supplied PDT at %r is properly installed' % pdt)
        return
    else:
        raise TauError('Invalid PDT directory %r' % pdt)
    
    # Banner
    LOGGER.info('Installing PDT at %r' % prefix)

    # Configure the source code for this configuration
    srcdir = PDT_SRC_DIR
    cmd = getConfigureCommand(config)
    LOGGER.debug('Creating configure subprocess in %r: %r' % (srcdir, cmd))
    LOGGER.info('Configuring PDT...')
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('PDT configure failed.')

    # Execute make
    cmd = ['make', '-j', 'install']
    LOGGER.debug('Creating make subprocess in %r: %r' % (srcdir, cmd))
    LOGGER.info('Installing PDT...')
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('PDT compilation failed.')
    
    # Clean up
    shutil.rmtree(srcdir)
    LOGGER.debug('Recursively deleting %r' % srcdir)
    LOGGER.info('PDT installation complete.')
