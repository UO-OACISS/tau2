# Copyright 2005-2013 Intel Corporation.  All Rights Reserved.
#
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its
# suppliers or licensors.  Title to the Material remains with Intel
# Corporation or its suppliers and licensors.  The Material is protected
# by worldwide copyright laws and treaty provisions.  No part of the
# Material may be used, copied, reproduced, modified, published, uploaded,
# posted, transmitted, distributed, or disclosed in any way without
# Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other
# intellectual property right is granted to or conferred upon you by
# disclosure or delivery of the Materials, either expressly, by
# implication, inducement, estoppel or otherwise.  Any license under such
# intellectual property rights must be express and approved by Intel in
# writing.

# set this to your TBB library location
TBB_HOME=/usr/local/tbb2018_20170919oss
INTEL_ARCH=intel64
CXXFLAGS=-I$(TBB_HOME)/include -L$(TBB_HOME)/lib/$(INTEL_ARCH)/gcc4.7 -Wl,-rpath,$(TBB_HOME)/lib/$(INTEL_ARCH)/gcc4.7 -g -fpermissive

# GNU Makefile that builds and runs example.
# Unwound profile
# run_cmd=TAU_EBS_UNWIND=1 TAU_EBS_KEEP_UNRESOLVED_ADDR=1 tau_exec -T tbb,icpc,serial -ebs
# Flat profile only
run_cmd=TAU_EBS_KEEP_UNRESOLVED_ADDR=1 tau_exec -v -T tbb,serial
PROG=tree_sum
ARGS=auto 100000000 silent
PERF_RUN_ARGS=auto 100000000 silent

# The C++ compiler
ifneq (,$(shell which icc 2>/dev/null))
endif # icc
CXX=tau_cxx.sh -optCompInst

ifeq ($(shell uname), Linux)
ifeq ($(target), android)
LIBS+= --sysroot=$(SYSROOT)
run_cmd=../../common/android.linux.launcher.sh
else
LIBS+= -lrt 
endif
endif

all:	release test

release: *.cpp
	$(CXX) -O2 -DNDEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbbmalloc -ltbb $(LIBS)

debug: *.cpp
	$(CXX) -O0 -g -DTBB_USE_DEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbbmalloc_debug -ltbb_debug $(LIBS)

clean:
	$(RM) $(PROG) *.o *.d profile.*

test:
	$(run_cmd) ./$(PROG) $(ARGS)

perf_build: release

perf_run:
	./$(PROG) $(PERF_RUN_ARGS)
