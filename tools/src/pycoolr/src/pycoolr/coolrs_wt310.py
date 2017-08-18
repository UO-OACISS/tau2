#!/usr/bin/env python
#
# a code to read Yokogawa wt310 via the usbtmc interface
#
# This code requires the usbtmc driver for temperature reading.
#
# /dev/usbtmc[0-9] needs to be readable for this code
#
# Contact: Kazutomo Yoshii <ky@anl.gov>
#

import re, os, sys
import time
import smq
import keypress
import json

class wt310_reader:

    def __init__(self):
        self.fd = -1

    def open(self, devfn = '/dev/usbtmc0'):
        self.fd = os.open(devfn, os.O_RDWR)
        if self.fd < 0:
            return -1

        return 0
    
    # read() and write() are low-level method
    def read(self):
        if self.fd < 0:
            return -1
        buf = os.read(self.fd, 256) # 256 bytes
        return buf

    def write(self, buf):
        if self.fd < 0:
            return -1
        n = os.write(self.fd, buf)
        return n

    # wt310 methods should call read() and write()
    def set(self, cmd):
        self.write(cmd)

    def get(self, cmd):
        self.write(cmd)
        buf = self.read()
        return buf

    # send wt310 commands

    def readvals(self):
        buf = self.get(":NUM:NORM:VAL?\n") # is \n required?
        return buf.split(',')


    def sample(self):
        a = self.readvals()
        # default setting. to query, :NUM:NORM:ITEM1? for exampe
        # 1: Watt hour, 2: Current, 3: Active Power, 4: Apparent Power 
        # 5: Reactive Power, 6: Power factor
        # a's index is the item no minus one

        ret = {}
        # ret['WH'] = float(a[0])
        ret['J'] = float(a[0]) * 3600.0
        ret['P']  = float(a[2])
        ret['PF'] = float(a[5])
        return ret

    def start(self):
        # set item1 watt-hour
        self.set(":NUM:NORM:ITEM1 WH,1")
        # start integration
        self.set(":INTEG:MODE CONT")
        self.set(":INTEG:START")
        # ":INTEG:RESET" reset integration
        # ":RATE 100MS"  set the update rate 100ms

    def reset(self):
        self.set('*RST')  # initialize the settings
        self.set(':COMM:REM ON') # set remote mode

    def stop(self):
        self.set(":INTEG:STOP")  # stop integration


import getopt

def usage():
    print ''
    print 'Usage: coolrs_wt310.py [options]'
    print ''
    print '[options]'
    print ''
    print '-i int : sampling interval in sec'
    print '-c str : command string'
    print '-o str : output filename'
    print '-s str : start the mq producer. str is ip address'
    print ''

if __name__ == '__main__':

    interval_sec = .5
    outputfn = ''
    smqflag = False
    ipaddr = ''

    cmdmode = ''
    cmd = ''
    samplemode = False

    shortopt = "hi:o:s:"
    longopt = ["set=", "get=", "sample"]
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   shortopt, longopt)
    except getopt.GetoptError, err:
        print err
        usage()
        sys.exit(1)

    for o, a in opts:
        if o in ('-h'):
            usage()
            sys.exit(0)
        elif o in ('-i'):
            interval_sec = float(a)
        elif o in ('--sample'):
            samplemode = True
        elif o in ('--set'):
            cmdmode = 'set'
            cmd = a
        elif o in ('--get'):
            cmdmode = 'get'
            cmd = a
        elif o in ('-o'):
            outputfn = a
        elif o in ('-s'):
            smqflag = True
            ipadr = a
        else:
            print 'Unknown:', o, a
            
    #
    #

    wt310 = wt310_reader()

    if wt310.open():
        sys.exit(1)

    if samplemode:
        s = wt310.sample()
        ts = time.time()
        str = '{"sample":"wt310", "time":%.2lf, "power":%.2lf}' % \
            (ts, s['P'])
        print str
        sys.exit(0)

    if len(cmd) > 0:
        if cmdmode == 'set':
            wt310.set(cmd)
        else:
            print wt310.get(cmd)
        sys.exit(0)

    f = sys.stdout

    if len(outputfn) > 0:
        try:
            f = open(outputfn, 'w')
        except:
            print 'Error: failed to open', fn
            sys.exit(1)

        print 'Writing to', outputfn

    cfg = {}
    cfg["c1"] = {"label":"Time","unit":"Sec"}
    cfg["c2"] = {"label":"Power", "unit":"Watt"}
    #cfg["c"] = {"label":"Power Factor", "unit":""}
    #cfg["c"] = {"label":"Energy","unit":"Joules"}


    print >>f, json.dumps(cfg)

    if smqflag:
        mq = smq.producer(ipaddr)
        mq.start()
        mq.dict = cfg
        print 'Message queue is started:', ipaddr

    kp = keypress.keypress()
    kp.enable()
    print 'Press "q" to terminate'

    while True:
        wt310.start()

        s = wt310.sample()

        ts = time.time()
        #str = '%.2lf %.0lf %.2lf %.4lf' % \
        #    (ts, s['J'], s['P'], s['PF'])
        str = '%.2lf %.2lf' % \
            (ts, s['P'])
        print >>f, str
        f.flush()

        if smqflag:
            mq.append(str)

        time.sleep(interval_sec)

        if kp.available() and kp.readkey() == 'q':
            break

    wt310.stop()
    kp.disable()

    print 'terminated.'
