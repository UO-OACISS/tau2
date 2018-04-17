#!/bin/bash -e

make clean
make

set -x
for t in *_test ; do
    tau_exec -T ompt,serial,param -ompt ${t}
done