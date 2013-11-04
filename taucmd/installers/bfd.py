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
from taucmd import TauError

LOGGER = taucmd.getLogger(__name__)


# Default BFD download URL
BFD_URL = 'http://www.cs.uoregon.edu/research/paracomp/tau/tauprofile/dist/binutils-2.23.2.tar.gz'
# For debugging
#BFD_URL = 'http://localhost:3000/binutils-2.23.2.tar.gz'

# User specific BFD source code location
BFD_SRC_DIR = os.path.join(taucmd.SRC_DIR, 'bfd')



def downloadSource(src=BFD_URL):
    """
    Download and extract a GNU Binutils archive
    """
    dest = os.path.join(taucmd.SRC_DIR, src.split('/')[-1])
    LOGGER.info('Downloading BFD from %r' % src)
    util.download(src, dest)
    extractSource(dest)
    os.remove(dest)


def extractSource(tgz):
    """
    Extract a GNU Binutils archive
    """
    bfd_path = util.extract(tgz, taucmd.SRC_DIR)
    shutil.rmtree(BFD_SRC_DIR, ignore_errors=True)
    shutil.move(bfd_path, BFD_SRC_DIR)

    
def getConfigureCommand(config):
    """
    Returns the command that will configure BFD for this project
    """
    arch = config['target-arch']
    prefix = config['bfd-prefix']
    if arch == 'bgp':
        return ['./configure', 'CFLAGS=-fPIC', 'CXXFLAGS=-fPIC', '--prefix=%s' % prefix,
                'CC=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-gcc',
                'CXX=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-g++',
                '--disable-nls', '--disable-werror']
    elif arch == 'bgq':
        return ['./configure', 'CFLAGS=-fPIC', 'CXXFLAGS=-fPIC', '--prefix=%s' % prefix,
                'CC=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-gcc',
                'CXX=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-g++',
                '--disable-nls', '--disable-werror']
    elif arch == 'rs6000' or arch == 'ibm64':
        return ['./configure', 'CFLAGS=-fPIC', 'CXXFLAGS=-fPIC', '--prefix=%s' % prefix,
                 '--disable-nls', '--disable-werror', '--disable-largefile']
    else:
        return ['./configure', 'CFLAGS=-fPIC', 'CXXFLAGS=-fPIC', '--prefix=%s' % prefix,
             '--disable-nls', '--disable-werror'] 


def getPrefix(config):
    # TODO: Support other compilers 
    prefix = os.path.join(taucmd.TAUCMD_HOME, 'bfd', 'GNU')
    return prefix


def install(config, stdout=sys.stdout, stderr=sys.stderr):
    """
    Installs BFD
    """
    bfd = config['bfd']
    prefix = getPrefix(config)
    if not bfd or os.path.isdir(prefix):
        LOGGER.debug("Skipping BFD installation: bfd=%r, prefix=%r" % (bfd, prefix))
        return
        
    # Download and extract PDT if needed
    if bfd == 'download':
        downloadSource()
    elif os.path.isfile(bfd):
        extractSource(bfd)
    elif os.path.isdir(bfd):
        LOGGER.debug('Assuming user-supplied BFD at %r is properly installed' % bfd)
        return
    else:
        raise TauError('Invalid BFD directory %r' % bfd)
    
    # Banner
    LOGGER.info('Installing BFD at %r' % prefix)

    # Configure the source code for this configuration
    srcdir = BFD_SRC_DIR
    cmd = getConfigureCommand(config)
    LOGGER.debug('Creating configure subprocess in %r: %r' % (srcdir, cmd))
    LOGGER.info('Configuring BFD...')
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    #proc = subprocess.Popen(' '.join(cmd), cwd=srcdir, stdout=stdout, stderr=stderr, shell=True)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('BFD configure failed.')

    # Execute make
    cmd = ['make']
    LOGGER.debug('Creating make subprocess in %r: %r' % (srcdir, cmd))
    LOGGER.info('Compiling BFD...')
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    #proc = subprocess.Popen(' '.join(cmd), cwd=srcdir, stdout=stdout, stderr=stderr, shell=True)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('BFD compilation failed.')

    # Execute make install
    cmd = ['make', 'install']
    LOGGER.debug('Creating make subprocess in %r: %r' % (srcdir, cmd))
    LOGGER.info('Installing BFD...')
    proc = subprocess.Popen(cmd, cwd=srcdir, stdout=stdout, stderr=stderr)
    #proc = subprocess.Popen(' '.join(cmd), cwd=srcdir, stdout=stdout, stderr=stderr, shell=True)
    if proc.wait():
        shutil.rmtree(prefix, ignore_errors=True)
        raise TauError('BFD compilation failed.')
    
    # Cleanup
    shutil.rmtree(srcdir)
    LOGGER.debug('Recursively deleting %r' % srcdir)
    LOGGER.info('BFD installation complete.')
