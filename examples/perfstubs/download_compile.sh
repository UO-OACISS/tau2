#!/bin/bash
set -x
set -e

git clone https://github.com/UO-OACISS/perfstubs
mkdir compile
cd compile
cmake -DPERFSTUBS_BUILD_EXAMPLES=TRUE ../perfstubs
make
cd ..
cp compile/examples/perfstubs_test_api_c_no_tool perfstubs_test_api_c_no_tool

