#! /bin/sh

# this script expects an input that contains at least three lines specifying
# the library interface version, e.g.
# library.current=1
# library.revision=2
# library.age=4

# order of current, revision, age in the input file does matter!
version="$(sed -n -e 's/library.current=\+// p'  \
                  -e 's/library.revision=\+// p' \
                  -e 's/library.age=\+// p'      \
           "$1")"
set -- $version
printf "%d:%d:%d" $1 $2 $3
