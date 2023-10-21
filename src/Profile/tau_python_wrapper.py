# -*- coding: utf-8 -*-
#
# Copyright (c) 2015, ParaTools, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# (1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# (2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# (3) Neither the name of ParaTools, Inc. nor the names of its contributors may
#     be used to endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""TAU python instrumentation wrapper.

usage: python -m tau_python_wrapper MODULE

Runs MODULE with automatic Python instrumentation.
"""

from __future__ import print_function
import os
import sys

def dieInFlames(msg):
  print(msg, file=sys.stderr)
  sys.exit(-1)

try:
  modname = sys.argv[1]
except IndexError:
  dieInFlames('Usage: %s <modulename>' % sys.argv[0])

try:
  import tau
except ImportError:
  dieInFlames("module 'tau' not found in PYTHONPATH")
except:
  dieInFlames("Unknown exception while importing tau: %s" % sys.exc_info()[0])

compile_time_python_ver = tau.get_ctau_python_version()
runtime_python_ver = sys.version_info[0:3]
if compile_time_python_ver != runtime_python_ver:
    dieInFlames("TAU Error: TAU was compiled against Python {}.{}.{}, but version {}.{}.{} was used at runtime. The compile and runtime versions of Python must be identical.".format(compile_time_python_ver[0], compile_time_python_ver[1], compile_time_python_ver[2], runtime_python_ver[0], runtime_python_ver[1], runtime_python_ver[2]))

if sys.argv[1] == '-c':
  # tau_python -c 'some python commmand'
  command = sys.argv[2]
  del sys.argv[2]
  del sys.argv[0]
  tau.run(command)   
elif sys.argv[1] == '-m':
  # tau_python -m moduleName.foo
  modname = sys.argv[2]
  try:
    import pkgutil
    pkg_loader = pkgutil.get_loader(modname)
  except Exception as e:
    dieInFlames("The name '{}' does not name a module in $PYTHONPATH or $PWD".format(modname))
  if pkg_loader is None:
    dieInFlames("The name '{}' does not name a module in $PYTHONPATH or $PWD".format(modname))
  # When python is run with -m, current directory is added to search path
  sys.path.append(os.getcwd())
  # Fix up argv to hide tau_python
  # Find out the path to the module we are launching
  filename = ""
  try:
    # New way of getting package filename
    filename = pkg_loader.get_filename()
  except Exception as e:
    # old way
    filename = pkg_loader.filename
  sys.argv[0] = filename
  # remove the args
  del sys.argv[1]
  del sys.argv[1]
  if os.path.exists(modname) and filename[-3:].lower() != '.py':
    tau.runmodule(sys.argv[0])  
  else:
    tau.runmoduledir(modname)
else:
  # If we launched a Python script using the normal method,
  # argv would have the path to the script in argv[0].
  # Fix argv to be as it would have been without tau_python
  if os.path.exists(modname):
    path = os.path.dirname(modname)
    modname = os.path.basename(modname)
    if modname[-3:].lower() != '.py':
      dieInFlames("Sorry, I don't know how to instrument '%s'" % modname)
    modname = modname[:-3]
    sys.path.append(path)
    sys.argv = sys.argv[1:]
  
  tau.runmodule(modname)

