#!/usr/bin/env python
#
# coolr rapl related codes
#
# This code requires the intel_powerclamp module.
#
# Contact: Kazutomo Yoshii <ky@anl.gov>
#

import os, sys, re, time


class rapl_reader:
    dryrun = False
    rapldir='/sys/devices/virtual/powercap/intel-rapl'
    # 
    # e.g.,
    # intel-rapl:0/name
    # intel-rapl:0/intel-rapl:0:0/name
    # intel-rapl:0/intel-rapl:0:1/name
    def __init__ (self):
        self.dirs = {}
        self.max_energy_range_uj_d = {}

        if self.dryrun :
            return 

        self.init = False
        if not os.path.exists(self.rapldir):
            return
        self.init = True

        for d1 in os.listdir(self.rapldir):
            dn = "%s/%s" % (self.rapldir,d1)
            fn = dn + "/name"
            if os.access( fn , os.R_OK ) :
                f = open( fn)
                l = f.readline().strip()
                f.close()
                if re.search('package-[0-9]+', l):
                    self.dirs[l] = dn
                    pkg=l
                    for d2 in os.listdir("%s/%s" % (self.rapldir,d1) ):
                        dn = "%s/%s/%s" % (self.rapldir,d1,d2)
                        fn = dn + "/name"
                        if os.access( fn, os.R_OK ) :
                            f = open(fn)
                            l = f.readline().strip()
                            f.close
                            if re.search('core|dram', l):
                                self.dirs['%s/%s' % (pkg,l)] = dn


        for k in sorted(self.dirs.keys()):
            fn = self.dirs[k] + "/max_energy_range_uj"
            try:
                f = open( fn )
            except:
                print 'Unable to open', fn
                sys.exit(0)
            self.max_energy_range_uj_d[k] = int(f.readline())
            f.close()

        self.start_energy_counter()

    def initialized(self):
        return self.init

    def shortenkey(self,str):
        return str.replace('package-','p')

#        for k in sorted(self.dirs.keys()):
#            print k, self.max_energy_range_uj_d[k]


    def readenergy(self):
        if not self.init:
            return

        ret = {}
        ret['time'] = time.time()
        if self.dryrun:
            ret['package-0'] = readuptime()*1000.0*1000.0
            return ret
        for k in sorted(self.dirs.keys()):
            fn = self.dirs[k] + "/energy_uj"
            v = -1
            for retry in range(0,10):
                try:
                    f = open( fn )
                    v = int(f.readline())
                    f.close()
                except:
                    continue
            ret[k] = v
        return ret

    def readpowerlimit(self):
        if not self.init:
            return

        ret = {}
        if self.dryrun:
            ret['package-0'] = 100.0
            return ret
        for k in sorted(self.dirs.keys()):
            fn = self.dirs[k] + '/constraint_0_power_limit_uw'
            v = -1
            for retry in range(0,10):
                try:
                    f = open( fn )
                    v = int(f.readline())
                    f.close()
                except:
                    continue
            ret[k] = v / (1000.0 * 1000.0) # uw to w
        return ret

    def diffenergy(self,e1,e2): # e1 is prev and e2 is not
        ret = {}
        ret['time'] = e2['time'] - e1['time']
        for k in self.max_energy_range_uj_d:
            if e2[k]>=e1[k]:
                ret[k] = e2[k] - e1[k]
            else:
                ret[k] = (self.max_energy_range_uj_d[k]-e1[k]) + e2[k]
        return ret

    # calculate the average power from two energy values
    # e1 and e2 are the value returned from readenergy()
    # e1 should be sampled before e2
    def calcpower(self,e1,e2): 
        ret = {}
        delta = e2['time'] - e1['time']  # assume 'time' never wrap around
        ret['delta']  = delta
        if self.dryrun:
            k = 'package-0'
            ret[k] = e2[k] - e1[k]
            ret[k] /= (1000.0*1000.0) # conv. [uW] to [W]
            return ret

        for k in self.max_energy_range_uj_d:
            if e2[k]>=e1[k]:
                ret[k] = e2[k] - e1[k]
            else:
                ret[k] = (self.max_energy_range_uj_d[k]-e1[k]) + e2[k]
            ret[k] /= delta
            ret[k] /= (1000.0*1000.0) # conv. [uW] to [W]
        return ret

    # this should be renamed to reset_...
    def start_energy_counter(self):
        if not self.initialized():
            return

        self.start_time_e = time.time()
        self.totalenergy = {}
        self.lastpower = {}

        e = self.readenergy()
        for k in sorted(e.keys()):
            if k != 'time':
                self.totalenergy[k] = 0.0
                self.lastpower[k] = 0.0
        self.prev_e = e

    # XXX: fix the total energy tracking later
    def read_energy_acc(self):
        if not self.initialized():
            return

        e = self.readenergy()

        de = self.diffenergy(self.prev_e, e)

        for k in sorted(e.keys()):
            if k != 'time':
                self.totalenergy[k] += de[k]
                self.lastpower[k] = de[k]/de['time']/1000.0/1000.0;
        self.prev_e = e

        return e

    def stop_energy_counter(self):
        if not self.initialized():
            return

        e = self.read_energy_acc()
        self.stop_time = time.time()

    def sample_and_json(self, label = "", accflag = False, node = ""):
        if not self.initialized():
            return

        e = self.readenergy()

        de = self.diffenergy(self.prev_e, e)

        for k in sorted(e.keys()):
            if k != 'time':
                if accflag:
                    self.totalenergy[k] += de[k]
                self.lastpower[k] = de[k]/de['time']/1000.0/1000.0;
        self.prev_e = e

        # constructing a json output
        s  = '{"sample":"energy","time":%.3f' % (e['time'])
        if len(node) > 0:
            s += ',"node":"%s"' % node
        if len(label) > 0:
            s += ',"label":"%s"' % label
        s += ',"energy":{'
        firstitem = True
        for k in sorted(e.keys()):
            if k != 'time':
                if firstitem:
                    firstitem = False
                else:
                    s+=','
                s += '"%s":%d' % (self.shortenkey(k), e[k])
        s += '},'
        s += '"power":{'

        totalpower = 0.0
        firstitem = True
        for k in sorted(self.lastpower.keys()):
            if k != 'time':
                if firstitem:
                    firstitem = False
                else:
                    s+=','
                s += '"%s":%.1f' % (self.shortenkey(k), self.lastpower[k])
                # this is a bit ad hoc way to calculate the total. needs to be fixed later
                if k.find("core") == -1:
                    totalpower += self.lastpower[k]
        s += ',"total":%.1f' % (totalpower)
        s += '},'

        s += '"powercap":{'
        rlimit = self.readpowerlimit()
        firstitem = True
        for k in sorted(rlimit.keys()):
            if firstitem:
                firstitem = False
            else:
                s+=','
            s += '"%s":%.1f' % (self.shortenkey(k), rlimit[k])
        s += '}'

        s += '}'
        return s

    def total_energy_json(self):
        if not self.initialized():
            return ''

        dt = self.stop_time - self.start_time_e
        # constructing a json output
        e = self.totalenergy
        s  = '{"total":"energy","difftime":%f' % (dt)
        for k in sorted(e.keys()):
            if k != 'time':
                s += ',"%s":%d' % (self.shortenkey(k), e[k])
        s += '}'
        return s


if __name__ == '__main__':

    rr = rapl_reader()

    if rr.initialized():
        rr.start_energy_counter()
        for i in range(0,3):
            time.sleep(1)
            s = rr.sample_and_json(accflag=True)
            print s
        rr.stop_energy_counter()
        s = rr.total_energy_json()
        print s

    else:
        print 'Error: No intel rapl sysfs found'
