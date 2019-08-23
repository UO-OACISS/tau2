#!/usr/bin/env python

import sys, os, re
import json
import time
import getopt

from listrotate import *
from clr_utils import *

# default values

cfgfn = ''
fakemode = False
appcfgfn = ''
targetnode = ''
enclaves = []
#intervalsec = 1.0
intervalsec = 0.5

# here is the priority: options > cfg > default
# cmd options are the highest priority

#print 'starting coolr-subscribe'

cfg = {}

cfg["outputfn"] = 'multinodes.json'
cfg["modnames"] = ['enclave', 'power', 'temp', 'runtime', 'freq', 'application']
cfg["figwidth"] = 20
cfg["figheight"] = 12
cfg["ncols"] = 3
cfg["nrows"] = 2

shortopt = "h"
# XXX: keep enclave= for compatibility
longopt = ['output=','node=', 'enclave=', 'enclaves=', 'width=', 'height=', 'list', 'mods=', 'ncols=', 'nrows=', 'appcfg=' ]
try:
    opts, args = getopt.getopt(sys.argv[1:],
                               shortopt, longopt)
except getopt.GetoptError, err:
    print err
    usage()
    sys.exit(1)

ocfg = {}
for o, a in opts:
    if o in ('-h'):
        usage()
        sys.exit(0)
    elif o in ("--node"):
        targetnode=a
    elif o in ("--appcfg"):
        appcfgfn=a
    elif o in ("--enclaves", "--enclave"):
        enclaves=a.split(',')
    elif o in ("--output"):
        ocfg["outputfn"]=a
    elif o in ("--width"):
        ocfg["figwidth"] = int(a)
    elif o in ("--height"):
        ocfg["figheight"] = int(a)
    elif o in ("--nrows"):
        ocfg["nrows"]=int(a)
    elif o in ("--ncols"):
        ocfg["ncols"]=int(a)
    elif o in ("--list"):
        print ''
        print '[available graph modules]'
        print ''
        for i in cfg["modnames"]:
            print i
        print ''
        print ''
        sys.exit(0)
    elif o in ("--mods"):
        ocfg["modnames"] = a.split(",")

if len(args) < 1:
    print ''
    print 'No config file is specified.  Enabled the fake mode.'
    print ''
    cfg["masternode"] = "frontend"
    cfg["drawexternal"] = "no"
    cfg["drawacpipwr"] = "no"
    cfg["dramrapl"] = "yes"
    cfg["tempmax"] = 90
    cfg["tempmax"] = 40
    cfg["freqmin"] = 0.8
    cfg["freqmax"] = 3.1
    cfg["freqnorm"] = 2.3
    cfg["pwrmax"] = 150
    cfg["pwrmin"] = 5
    cfg["acpwrmax"] = 430
    fakemode = True
else:
    cfgfn = args[0]
    #
    # load config files
    #
    with open(cfgfn) as f:
        cfgtmp = json.load(f)
        # override if cfg defines any
        for k in cfgtmp.keys():
            cfg[k] = cfgtmp[k]
        # override if specifed as cmd option
        for k in ocfg.keys():
            cfg[k] = ocfg[k]

if len(targetnode) == 0 :
    targetnode = cfg['masternode']
if len(enclaves) == 0:
    if cfg.has_key('enclaves'):
        enclaves = cfg['enclaves']

print 'masternode:', cfg['masternode']
print 'targetnode:', targetnode
print 'enclaves:', enclaves

if len(appcfgfn) > 0:
    with open(appcfgfn) as f:
        appcfg = json.load(f)
    for k in appcfg.keys():
        cfg[k] = appcfg[k]

    if not (cfg.has_key('appname') and cfg.has_key('appsamples')):
        print "Please double check %s: appname or appsamples tags" % appcfgfn
        sys.exit(1)


if fakemode:
    import fakedata
    targetnode='v.node'
    enclaves = ['v.enclave.1', 'v.enclave.2']
    info = json.loads(fakedata.gen_info(targetnode))
else:
    info = querydataj(cfg['queryinfocmd'])[0]
    
#
#
#
try:
    logf = open(cfg["outputfn"], 'w', 0) # unbuffered write
except:
    print 'unable to open', cfg["outputfn"]

print >>logf, json.dumps(info)

#if not fakemode:
#    querycmds = cfg['querycmds']


npkgs=info['npkgs']
lrlen=200  # to option
gxsec=120 # graph x-axis sec

#info = querydataj(cfg['queryinfocmd'])[0]

params = {}  # graph params XXX: extend for multinode
params['cfg'] = cfg
params['info'] = info
params['lrlen'] = lrlen
params['gxsec'] = gxsec
params['cur'] = 0  # this will be updated
params['pkgcolors'] = [ 'blue', 'green' ] # for now
params['targetnode'] = targetnode
params['enclaves'] = enclaves
