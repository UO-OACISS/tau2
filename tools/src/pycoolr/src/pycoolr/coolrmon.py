#!/usr/bin/env python

#
# coolr monitoring tool
#
# Contact: Kazutomo Yoshii <ky@anl.gov>

import sys, os, time
from collections import deque

#
import smq


if __name__ == '__main__':

    # default values
    intervalsec = 1.0
    maxdq = 360
    port = 23458


    if len(sys.argv) < 2:
        print 'Usage: %s p [port]'
        print ''
        sys.exit(1)


    ip = sys.argv[1]

    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    print 'ip:', ip
    print 'port:', port


    wt310 = smq.consumer(ip, port)
    d = wt310.get({'cmd':'cfg'})
    if len(d) == 0:
        print 'Please check the wt310 producer side!'
        sys.exit(1) # for now. implement a nicer code later


    d = wt310.get({'cmd':'clear'})

    while True:
        d = mqc.get({'cmd':'item'})
        if len(d) > 0:
            print 'item:', d['item'], 'remain:', d['len']

        time.sleep(intervalsec)

    print 'done'
