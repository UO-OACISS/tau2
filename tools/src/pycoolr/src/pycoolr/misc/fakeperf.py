#!/usr/bin/env python

import os, sys, re, time
import numpy as np
import subprocess
import random as r

debuglevel = 1

# XXX: tentative
class log2beacon:
    def __init__(self, cmd, topic):
        self.cmd = cmd
        self.topic = topic

    def logger(self,str):
        c = [self.cmd, self.topic, "%s" % str]
        try:
            subprocess.call(c)
        except:
            print 'Error: failed to publish:', c
            sys.exit(1)
        if debuglevel > 0:
            print 'Debug:', c

def gen_argobots_json():
    n = 20
    t = time.time()
    buf  = '{'
    buf += '"time":%lf,' % t
    buf += '"node":"frontend",'
    buf += '"sample":"argobots",'
    buf += '"num_es":%d,' % n
    buf += '"num_threads":{'
    for i in range(0,n-1):
        buf += '"es%d":%lf,' % (i, r.random()*100)
    buf += '"es%d":%lf}}' % (n-1, r.random()*100)
    
    return buf

def gen_appperf_json():
    t = time.time()
    buf  = '{'
    buf += '"time":%lf,' % t
    buf += '"node":"frontend",'
    buf += '"sample":"appperf",'
    buf += '"val":%lf}' %  (r.random() * 20.0)

    return buf

if __name__ == '__main__':
    l = log2beacon( '/nfs/beacon_inst/bin/arbitrary_pub', 'NODE_POWER')

    while True:
        l.logger( gen_appperf_json() )
        l.logger( gen_argobots_json() )

        time.sleep(1)



