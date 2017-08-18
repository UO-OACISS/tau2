#!/usr/bin/env python
#
# misc. classes, functions
#
# Contact: Kazutomo Yoshii <ky@anl.gov>
#

import os, sys, re, time

def readbuf(fn):
    for retry in range(0,10):
        try:
            f = open( fn )
            l = f.readline()
            f.close()
            return l
        except:
            time.sleep(0.01)
            continue
    return ''

def readuptime():
    f = open( '/proc/uptime' ) 
    l = f.readline()
    v = l.split()
    return float( v[0] )
