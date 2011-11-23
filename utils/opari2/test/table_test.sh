#!/bin/sh
TOOL=..
#set -x

${TOOL}/opari -table opari.tab.c || exit
diff out.oparirc.$1 opari.rc || exit
diff opari.tab.c opari.tab.$1.out || exit
