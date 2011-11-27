#! /bin/sh

# this script expects an input that contains at least two lines specifying
# the package version, e.g.
# package.major=0
# package.minor=9

# order of major and minore in the input file does matter!
version="$(sed -n -e 's/package.major=\+// p' \
                  -e 's/package.minor=\+// p' \
           "$1")"
set -- $version
printf "%d.%s" $1 $2
