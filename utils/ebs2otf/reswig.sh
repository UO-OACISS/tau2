#!/bin/bash

TAU_EBS_OTF_LIB=$1

swig -perl5 ebs2otf.i
gcc -fPIC -c ebs2otf_wrap.c    `perl -MExtUtils::Embed -e ccopts`
gcc -shared -o ebs2otf.so ebs2otf_wrap.o -L${TAU_EBS_OTF_LIB} -lotf -lz
