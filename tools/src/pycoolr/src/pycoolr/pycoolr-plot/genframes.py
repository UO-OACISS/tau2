#!/usr/bin/env python

#
# Reconstructing graph from recorded data by "realtime plot"
#
# Currently genframes only works with offline data
#
# Kazutomo Yoshii <ky@anl.gov>
# 

import os, sys, time, re
import json
import numpy as np
import math

class genframes:

    def __init__(self, jsonfn):
        # key is nodename
        self.info = {}
        self.data = {}

        # load all lines at init (naive)
        for jfn in jsonfn:
            with open(jfn) as f:
                while True:
                    l = f.readline()
                    if not l:
                        break
                    j = json.loads(l)
                    if j.has_key("nodeinfo"):
                        # one info per node, otherwise unknown
                        self.info[j["nodeinfo"]] = j
                    elif j.has_key("sample"):
                        nname = j["node"]
                        stype = j["sample"]

                        if not self.data.has_key(nname) :
                            self.data[nname] = {}
                        if not self.data[nname].has_key(stype) :
                            self.data[nname][stype] = []
                        self.data[nname][stype].append(j)
        # double check info's keys and data's keys match
        for k in sorted(self.info.keys()):
            if not self.data.has_key(k):
                raise KeyError

    def getnodes(self):
        n = []
        for k in self.data.keys():
            n.append(k)
        return n

    def getsamples(self,nname):
        s = []
        for k in self.data[nname].keys():
            s.append(k)
        return s
    
    def gettimerange(self):
        t0 = [] # start times
        t1 = [] # end times
        for k in self.data.keys():
            for k2 in self.data[k].keys():
                a = self.data[k][k2]
                t0.append(a[0]["time"])
                t1.append(a[-1]["time"])
        return np.max(t0), np.min(t1)

    # nth frame start time. call after setfps()
    def getnthstart(self,nth):
        v = self.ts + nth*self.interval
        return math.fmod(v, self.interval)
    
    def setfps(self,fps):
        self.ts, self.te = self.gettimerange()
        self.fps = fps
        self.interval = 1.0/fps
        self.nframes =  int((self.te-self.ts)/self.interval) - 1
        # index for each samples. (may not be used)
        self.sampleidx = {}
        for n in self.data.keys():
            self.sampleidx[n] = {}
            for s in self.data[n].keys():
                self.sampleidx[n][s] = -1
                data = self.data[n][s]
                for idx in range(len(data)):
                    t = data[idx]["time"]
                    if t >= self.ts and t < self.ts+self.interval:
                        self.sampleidx[n][s] = idx
                        break

    def gettime2frameno(self,t):
        n = int((t-self.ts)/self.interval)
        return n
                    
    def getlist(self,nname,sname):
        ret = [None for i in range(self.nframes)]
        data = self.data[nname][sname]
        for idx in range(len(data)):
            t = data[idx]["time"]
            fno = self.gettime2frameno(t)
            if fno >= 0 and fno < self.nframes:
                ret[fno] = data[idx]

        return ret

    
if __name__ == "__main__":

    r = genframes("testdata/chameleon.json")

    nodes = r.getnodes()
    print nodes

    ts,te = r.gettimerange()

    r.setfps(4)

    l = r.getlist(nodes[0], 'temp')
    for i in l:
        if i == None:
            print 'na'
        else:
            print i['time']
