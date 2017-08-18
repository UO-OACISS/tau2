#!/usr/bin/env python

#
# coolr monitoring library
#
# the following kernel modules must be installed
#
# - coretemp
# - intel_powerclamp
#
# the following kernel modules are optional
#
# - acpi_power_meter
# - cpustat (coolr-hpc)
#
# written by Kazutomo Yoshii <ky@anl.gov>
#

import os, sys, re, time
import numpy as np
import subprocess
import threading
import signal

# local
import keypress
import clr_rapl
import clr_hwmon
import clr_nodeinfo
#import clr_amperf
import clr_cpufreq

from clr_misc import *

#
#

debuglevel=0

#
#

import getopt

def usage():
    print ''
    print 'Usage: coolrmon.py [options] cmd args..'
    print ''
    print '[options]'
    print ''
    print '--runtests : run internal test funcs'
    print '-C or --cooldown temp : wait until the max coretemp gets below temp'
    print ''

class coolrmon_tracer:
    def sample_acpi(self, label):
        if self.acpi.initialized():
            s = self.acpi.sample_and_json(node=self.nc.nodename)
            self.logger(s)

    def sample_freq(self,label):
        #s = self.amp.sample_and_json()
        if self.freq.init:
            s = self.freq.sample_and_json(node=self.nc.nodename)
            self.logger(s)

    def sample_temp(self,label):
        s = self.ctr.sample_and_json(node=self.nc.nodename)
        self.logger(s)

    def sample_energy(self, label):

        accflag = False
        if label == 'run' :
            accflag = True
        s = self.rapl.sample_and_json(label, accflag,\
                                          node=self.nc.nodename)
        self.logger(s)

    def cooldown(self,label):
        if self.cooldowntemp < 0:
            return

        while True: 
            self.sample_energy(label)
            self.sample_temp(label)

            # currently use only maxcoretemp
            # XXX: readtempall() is called from sample_temp()
            # possible to optimize this
            temp = self.ctr.readtempall()
            maxcoretemp = self.ctr.getmaxcoretemp(temp)
            if maxcoretemp < self.cooldowntemp:
                break
            time.sleep(self.intervalsec)

    def setcooldowntemp(self,v):
        self.cooldowntemp = v

    def setinterval(self,isec):
        self.intervalsec = isec 

    def setbeacon(self):
        self.beacon = True
        self.ctr.outputpercore(False)
        self.freq.outputpercore(False)

    def setlogger(self,func):
        self.logger = func

    def defaultlogger(self,str):
        print str

    def __init__ (self):
        # default values
        self.cooldowntemp = 45  # depend on arch
        #self.output = sys.stdout
        self.intervalsec = 1
        #XXX: demo
        self.beacon = False
        self.logger = self.defaultlogger
        # instantiate class
        self.nc = clr_nodeinfo.nodeconfig()
        self.ct = clr_nodeinfo.cputopology()

        self.samples = []
        self.ctr = clr_hwmon.coretemp_reader()
        self.samples.append("temp") # XXX: fix coretemp_reader.init
        self.rapl = clr_rapl.rapl_reader()
        if self.rapl.init:
            self.samples.append("energy")
        #self.amp = clr_amperf.amperf_reader()
        self.freq =  clr_cpufreq.cpufreq_reader()
        if self.freq.init:
            self.samples.append("freq")
        self.acpi = clr_hwmon.acpi_power_meter_reader()
        if self.acpi.init:
            self.samples.append("acpi")

    def shownodeinfo(self):
        s  = '{"nodeinfo":"%s"' % self.nc.nodename
        s += ',"kernelversion":"%s"' % self.nc.version
        s += ',"cpumodel":%d' % self.nc.cpumodel
        s += ',"memoryKB":%d' % self.nc.memoryKB
        s += ',"freqdriver":"%s"' % self.nc.freqdriver
        s += ',"samples":[%s]' % \
             ','.join(map((lambda s:'"%s"'%s), self.samples))
        ncpus = len(self.ct.onlinecpus)
        s += ',"ncpus":%d' % ncpus
        npkgs = len(self.ct.pkgcpus.keys())
        s += ',"npkgs":%d' % npkgs

        for p in sorted(self.ct.pkgcpus.keys()):
            s += ',"pkg%d":[' % p
            s += ','.join(map(str,self.ct.pkgcpus[p]))
            s += ']'

        for p in sorted(self.ct.pkgcpus.keys()):
            s += ',"pkg%dphyid":[' % p
            phyid = []
            for cpuid in self.ct.pkgcpus[p]:
                phyid.append(self.ct.cpu2coreid[cpuid][1])
            s += ','.join(map(str,phyid))
            s += ']'

        s += ',"nnodes":%d' % len(self.ct.nodecpus.keys())
        for n in sorted(self.ct.nodecpus.keys()):
            s += ',"node%d":[' % n
            s += ','.join(map(str,self.ct.nodecpus[n]))
            s += ']'

        if self.rapl.initialized():
            s += ',"max_energy_uj":{'
            firstitem = True
            for p in sorted(self.ct.pkgcpus.keys()):
                k = 'package-%d' % p  # XXX: double check both 'topology' and rapldriver use the same numbering scheme.
                if firstitem :
                    firstitem = False
                else:
                    s += ','
                s += '"p%d":%d' % (p, self.rapl.max_energy_range_uj_d[k])
            s += '}'
        s += '}'

        self.logger(s)


    def tee(self,argv):
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE )

        while True:
            l = proc.stdout.readline()
            if not l:
                break
            print 'stdout:', l,


    def sigh(signal, frame):
        self.kp.disable()
        sys.exit(1)

    def run(self, argv):
        self.shownodeinfo()

        self.start_time0 = time.time()
        s = '{"starttime":%.3f}' % self.start_time0
        self.logger(s)

        if len(argv) == 0:
            signal.signal(signal.SIGINT, self.sigh)
            signal.signal(signal.SIGTERM, self.sigh)

            self.kp = keypress.keypress()
            self.kp.enable()
            self.rapl.start_energy_counter()
            while True:
                self.sample_temp('run')
                self.sample_energy('run')
                self.sample_freq('run')
                self.sample_acpi('run')

                time.sleep(self.intervalsec)
                if self.kp.available():
                    if self.kp.readkey() == 'q':
                        break
            self.rapl.stop_energy_counter()
            s = self.rapl.total_energy_json()
            self.logger(s)
            self.kp.disable()
            return

        self.cooldown('cooldown')

        # spawn a process and start the sampling thread
        t = threading.Thread(target=self.tee, args=[argv])
        t.start()

        self.rapl.start_energy_counter()

        while True:
            if not t.isAlive():
                break

            self.sample_temp('run')
            self.sample_energy('run')
            self.sample_freq('run')
            self.sample_acpi('run')

            time.sleep(self.intervalsec)

        self.rapl.stop_energy_counter()
        s = self.rapl.total_energy_json()
        self.logger(s)

        self.cooldown('cooldown')

class log2file:
    def __init__(self, fn):
        try:
            self.f = open(fn, 'w')
        except:
            print 'Error: failed to open', fn
            self.f = sys.stdout

    def logger(self,str):
        self.f.write(str + '\n')

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
            time.sleep(5)
        if debuglevel > 0:
            print 'Debug:', c

if __name__ == '__main__':

    shortopt = "hC:i:o:"
    longopt = ['runtests', 'cooldown=', 'interval=', 'output=', 'sample', 'info', 'output2beacon=', 'debug=' ]
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   shortopt, longopt)

    except getopt.GetoptError, err:
        print err
        usage()
        sys.exit(1)

    cmd = ''
    tr = coolrmon_tracer()

    for o, a in opts:
        if o in ('-h'):
            usage()
            sys.exit(0)
        elif o in ("--debug"):
            debuglevel=int(a)
        elif o in ("-C", "--cooldown"):
            tr.setcooldowntemp(int(a))
        elif o in ("-i", "--interval"):
            tr.setinterval(float(a))
        elif o in ("-o", "--output"):
            l = log2file(a)
            tr.setlogger(l.logger)
        elif o in ("--output2beacon"):
            cmd, topic = a.split(':')
            if len(topic) == 0:
                print 'Specify topic after :'
                sys.exit(1)
            l = log2beacon(cmd, topic)
            tr.setlogger(l.logger)
            tr.setbeacon()
        elif o in ("--sample"):
            tr.sample_temp('sample')
            tr.sample_energy('sample')
            tr.sample_freq('sample')
            tr.sample_acpi('sample')
            sys.exit(0)
        elif o in ("--info"):
            tr.shownodeinfo()
            sys.exit(0)
        else:
            print 'Unknown:', o, a
            sys.exit(1)
    #
    #
    # XXX: beacon is ad hoc impl for the demo
    if tr.beacon:
        while True:
            tr.sample_temp('run')
            tr.sample_freq('run')
            time.sleep(tr.intervalsec)
    else:
        tr.run(args)

    sys.exit(0)
