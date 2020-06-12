#
# Copyright 2015 NVIDIA Corporation. All rights reserved
#

# CUPTI OpenACC support is only available on Linux x86_64 and ppc64le.

# Point to your PGI OpenACC installation's include directory.
# PGI compiler must be in PATH.
NVCC:=$(shell which nvcc)
PGCC:=$(shell which pgcc)
BINDIR:=$(dir $(NVCC))
BINDIR2:=$(dir $(PGCC))
CUDADIR:=$(BINDIR)..
#CUPTIDIR=$(CUDADIR)/extras/CUPTI
CUPTIDIR=$(CUDA)/extras/CUPTI
WORKDIR:=$(shell pwd)

OPENACC_INCLUDE_PATH:=$(BINDIR2)..

HOST_ARCH := $(shell uname -m)

INCLUDES += -I$(CUPTIDIR)/include
INCLUDES += -I$(CUDADIR)/include
INCLUDES += -I$(OPENACC_INCLUDE_PATH)

PGCPP       = pgc++
PGCPP_FLAGS = -acc -ta=tesla:cuda9.2 -Mcuda=nordc -pgf90libs
ifneq ($(HOST_ARCH),ppc64le)
        PGCPP_FLAGS += -tp=p7 -mp=nonuma
else
        PGCPP_FLAGS += -DHOST_ARCH_PPC=1
endif

export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(CUPTIDIR)/lib64

all: openacc_app

openacc_app: openacc_app.o dataElem_um.o
	$(PGCPP) $(PGCPP_FLAGS) -L. -L$(CUPTIDIR)/lib64 -o $@ $^ -Wl,-rpath,$(WORKDIR)

openacc_app.o: openacc_app.cpp
	$(PGCPP) $(INCLUDES) $(PGCPP_FLAGS) -c -o $@ $^

dataElem_um.o: dataElem_um.cu
	nvcc -ccbin pgc++ -arch=sm_70 $(INCLUDES) -c -o $@ $^

clean:
	rm -f openacc_app profile.* *.o

