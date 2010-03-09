#!/bin/bash

swig -perl5 otf.i
gcc -fPIC -c otf_wrap.c    `perl -MExtUtils::Embed -e ccopts`
gcc -shared -o otf.so otf_wrap.o -L/usr/local/packages/otf-1.6.4/lib -lotf -lz
