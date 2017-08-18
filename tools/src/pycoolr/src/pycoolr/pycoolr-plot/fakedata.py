#!/usr/bin/env python

import os, sys, re, time
import numpy as np
import subprocess
import random as r
import json

pkg0cpuid=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]
pkg1cpuid=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47]
pkg0phyid=[0,1,2,3,4,5,8,9,10,11,12,13,0,1,2,3,4,5,8,9,10,11,12,13]
pkg1phyid=[0,1,2,3,4,5,8,9,10,11,12,13,0,1,2,3,4,5,8,9,10,11,12,13]


def gen_info(node):
    buf  = '{"nodeinfo":"%s","kernelversion":"4.1.3-argo","cpumodel":63,"memoryKB":131787416,"freqdriver":"pstate",'
    buf += '"samples":["temp","energy","freq"],"ncpus":48,"npkgs":2,'
    buf += '"pkg0":[%s],' % (','.join(map(str,pkg0cpuid)))
    buf += '"pkg1":[%s],' % (','.join(map(str,pkg1cpuid)))
    buf += '"pkg0phyid":[%s],' % (','.join(map(str,pkg0phyid)))
    buf += '"pkg1phyid":[%s],' % (','.join(map(str,pkg1phyid)))
    buf += '"nnodes":2}'
    return buf

def gen_argobots(node):
    n = int(r.random()*10) + 20
    t = time.time()
    buf  = '{'
    buf += '"time":%lf,' % t
    buf += '"node":"%s",' % node
    buf += '"sample":"argobots",'
    buf += '"num_es":%d,' % n
    buf += '"num_threads":{'
    for i in range(0,n-1):
        buf += '"es%d":%lf,' % (i, r.random()*100)
    buf += '"es%d":%lf},' % (n-1, r.random()*100)
    buf += '"num_tasks":{'
    for i in range(0,n-1):
        buf += '"es%d":%lf,' % (i, r.random()*100)
    buf += '"es%d":%lf}}' % (n-1, r.random()*100)
    return buf

def gen_application(node):
    t = time.time()
    gen_application.cnt += 1
    a = (gen_application.cnt % 60)
    if not a in [i for i in range(5,25,2)]:
        return ""
    buf  = '{'
    buf += '"time":%lf,' % t
    buf += '"node":"%s",' % node
    buf += '"sample":"application",'
    buf += '"#TE_per_sec_per_node":%lf,' %  (r.random() * 100000.0)
    buf += '"#TE_per_watt_per_node":%lf,' %  (r.random() * 100000.0)
    buf += '"#TE_per_sec":%lf}' %  (r.random() * 100000.0)
    return buf
gen_application.cnt  = 0


def gen_rapl(node):
    t = time.time()
    buf  = '{'
    buf += '"node":"%s",' % node
    buf += '"sample":"energy",'
    buf += '"time":%lf,' % t
    buf += '"powercap":{"p0":120.0,"p1":120.0,"p0/dram":0.0,"p1/dram":0.0},'
    p1=r.random()*60.0 + 50
    p2=r.random()*60.0 + 60
    #if r.random() < 0.1:
    #p2 = -100
    p1d=r.random()*20.0 + 10
    p2d=r.random()*20.0 + 10
    buf += '"power":{"total":%.1lf,"p0":%.1lf,"p1":%.1lf,"p0/dram":%.1lf,"p1/dram":%.1lf}}' %\
           (p1+p2, p1, p2, p1d, p2d)

    # "energy": {"p0": 34, "p1": 34, "p0/dram": 25, "p1/dram": 25},
    return buf


def gen_enclave(node, off):
    t = time.time()
    buf  = '{'
    buf += '"node":"%s",' % node
    buf += '"sample":"energy",'
    buf += '"time":%lf,' % t
    buf += '"powercap":{"p0":120.0,"p1":120.0,"p0/dram":0.0,"p1/dram":0.0},'
    p1=r.random()*60.0 + 50 + off
    p2=r.random()*60.0 + 60 + off
    #if r.random() < 0.1:
    #p2 = -100
    p1d=r.random()*20.0 + 10
    p2d=r.random()*20.0 + 10
    buf += '"power":{"total":%.1lf,"p0":%.1lf,"p1":%.1lf,"p0/dram":%.1lf,"p1/dram":%.1lf}}' %\
           (p1+p2, p1, p2, p1d, p2d)

    # "energy": {"p0": 34, "p1": 34, "p0/dram": 25, "p1/dram": 25},
    return buf


def gen_mean_std(node,sample):
    t = time.time()
    buf  = '{'
    buf += '"node":"%s",' % node
    buf += '"sample":"%s",' % sample
    buf += '"time":%lf,' % t
    buf += '"p0":{"mean":%lf,"std":%lf},' % (r.random()*(30+40), r.random()*7)
    buf += '"p1":{"mean":%lf,"std":%lf}}' % (r.random()*(30+50), r.random()*5)
    return buf

def gen_freq(node):
    t = time.time()
    gen_freq.cnt += 1

    buf  = '{'
    buf += '"node":"%s",' % node
    buf += '"sample":"freq",'
    buf += '"time":%lf,' % t
    fn = []
    fs = []
    a = (gen_freq.cnt/5) % 3
    for i in pkg0cpuid:
        if a == 0:
            f = r.random()*0.4
        elif a == 1:
            if (i%2)==0:
                f = r.random()*2.3
            else:
                f = r.random()*0.4
        elif a == 2:
            f = r.random()*3.1

        fn.append(f)
        fs.append('"c%d":%.1lf' % (i,f))
    m = np.mean(fn)
    s = np.std(fn)
    buf += '"p0":{"mean":%lf,"std":%lf,%s},' % (m, s, ','.join(fs))

    fn = []
    fs = []
    for i in pkg0cpuid:
        f = r.random()*2.3 + 0.3
        fn.append(f)
        fs.append('"c%d":%.1lf' % (i,f))
    m = np.mean(fn)
    s = np.std(fn)
    buf += '"p1":{"mean":%lf,"std":%lf,%s}}' % (m, s, ','.join(fs))

    return buf
gen_freq.cnt = 0

def queryfakedataj():
    node="v.node"
    ret = []
    ba = [ gen_enclave("v.enclave.1", 0),
           gen_enclave("v.enclave.2", -30),
           gen_rapl(node),
           gen_mean_std(node,"temp"),
           gen_freq(node),
           gen_argobots(node),
           gen_application(node) ]

    for b in ba:
        if len(b) > 0:
            ret.append(json.loads(b))
    return ret

if __name__ == '__main__':

    j = queryfakedataj()
    for e in j:
        print e
