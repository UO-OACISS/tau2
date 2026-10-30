#!/bin/bash -e

set -x
spack install saltfm
spack find -p saltfm
