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

TAU_CC = 'tau_cc.sh'
TAU_CXX = 'tau_cxx.sh'
TAU_F77 = 'tau_f90.sh'     # Yes, f90 not f77
TAU_F90 = 'tau_f90.sh'
TAU_UPC = 'tau_upc.sh'

class Compiler:
    pass

class C_Compiler(Compiler):
    TAU_COMMAND = TAU_CC
    
class CXX_Compiler(Compiler):
    TAU_COMMAND = TAU_CXX
    
class F77_Compiler(Compiler):
    TAU_COMMAND = TAU_F77

class F90_Compiler(Compiler):
    TAU_COMMAND = TAU_F90
    
class UPC_Compiler(Compiler):
    TAU_COMMAND = TAU_UPC

class GNU_CC(C_Compiler):
    NAME = 'GNU C Compiler'
    COMMANDS = ['gcc']
    VERSION_FLAG = '--version'
    VERSION_REGEX = '\(GCC\)\s+(\d+)\.(\d+).(\d+)'

class GNU_CXX(CXX_Compiler):
    NAME = 'GNU C++ Compiler'
    COMMANDS = ['g++']
    VERSION_FLAG = '--version'
    VERSION_REGEX = '\(GCC\)\s+(\d+)\.(\d+).(\d+)'

class GNU_F77(F77_Compiler):
    NAME = 'GNU Fortran Compiler'
    COMMANDS = ['gfortran']
    VERSION_FLAG = '--version'
    VERSION_REGEX = '\(GCC\)\s+(\d+)\.(\d+).(\d+)'

class GNU_F90(F90_Compiler):
    NAME = 'GNU Fortran Compiler'
    COMMANDS = ['gfortran']
    VERSION_FLAG = '--version'
    VERSION_REGEX = '\(GCC\)\s+(\d+)\.(\d+).(\d+)'
    
class GNU_UPC(UPC_Compiler):
    NAME = 'GNU UPC Compiler'
    COMMANDS = ['gupc']
    VERSION_FLAG = '--version'
    VERSION_REGEX = '\(GCC\)\s+(\d+)\.(\d+).(\d+)'
    
class MPI_CC(Compiler):
    NAME = 'MPI C Compiler Wrapper'
    COMMANDS = ['mpicc']
    VERSION_FLAG = '--version'
    VERSION_REGEX = None
    
class MPI_CXX(Compiler):
    NAME = 'MPI C++ Compiler Wrapper'
    COMMANDS = ['mpicxx', 'mpic++']
    VERSION_FLAG = '--version'
    VERSION_REGEX = None
    
class MPI_F77(Compiler):
    NAME = 'MPI FORTRAN77 Compiler Wrapper'
    COMMANDS = ['mpif77']
    VERSION_FLAG = '--version'
    VERSION_REGEX = None

class MPI_F90(Compiler):
    NAME = 'MPI Fortran90 Compiler Wrapper'
    COMMANDS = ['mpif90']
    VERSION_FLAG = '--version'
    VERSION_REGEX = None

GNU_COMPILERS = [GNU_CC, GNU_CXX, GNU_F77, GNU_F90, GNU_UPC]

MPI_COMPILERS = [MPI_CC, MPI_CXX, MPI_F77, MPI_F90]

ALL_COMPILERS = [cc for family in [GNU_COMPILERS, MPI_COMPILERS] for cc in family]

def identify(command):
    for cc in ALL_COMPILERS:
        if command in cc.COMMANDS:
            return cc