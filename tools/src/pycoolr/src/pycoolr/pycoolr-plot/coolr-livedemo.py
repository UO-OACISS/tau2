#!/usr/bin/env python

#
# matplot-based live demo tool, customized for the Argo demo
#
# Contact: Kaz Yoshii <ky@anl.gov>
#

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

cfg = {}

cfg["outputfn"] = 'multinodes.json'
cfg["modnames"] = ['enclave', 'power', 'temp', 'runtime', 'freq', 'application']
cfg["figwidth"] = 20
cfg["figheight"] = 12
cfg["ncols"] = 3
cfg["nrows"] = 2


def usage():
    print ''
    print 'Usage: %s [options] config_file' % sys.argv[0]
    print ''
    print '[options]'
    print ''
    print '--interval sec: specify the interval in sec. no guarantee. (default: %.1f)' % intervalsec
    print ''
    print '--outputfn fn : specify output fiflename (default: %s)' % cfg["outputfn"]
    print ''
    print '--enclaves CSV : enclave names. comma separated values without space'
    print '--node  name : target node the node power, temp, freq and app graphs'
    print ''
    print '--width int  : the width of the entire figure (default: %d)' % cfg["figwidth"]
    print '--height int : the height of the entire figure (default: %d)' % cfg["figheight"]
    print ''
    print '--ncols : the number of columns (default: %s)' % cfg["ncols"]
    print '--nrows : the number of rows (default: %s)' % cfg["nrows"]
    print ''
    print '--appcfg cfg : add-on config for application specific values'
    print ''
    print '--list : list available graph module names'
    print '--mods CSV : graph modules. comma separated values wihtout space'
    print ''

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

if not fakemode:
    querycmds = cfg['querycmds']


npkgs=info['npkgs']
lrlen=200  # to option
gxsec=120 # graph x-axis sec

#
#
#
params = {}  # graph params XXX: extend for multinode
params['cfg'] = cfg
params['info'] = info
params['lrlen'] = lrlen
params['gxsec'] = gxsec
params['cur'] = 0  # this will be updated
params['pkgcolors'] = [ 'blue', 'green' ] # for now
params['targetnode'] = targetnode
params['enclaves'] = enclaves


#
# matplot related modules
#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
#matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.size': 8})
from clr_matplot_graphs import *

fig = plt.figure( figsize=(cfg["figwidth"],cfg["figheight"]) )
fig.canvas.set_window_title('COOLR live demo tool')

plt.ion()
plt.show()

class layoutclass:
    def __init__(self, row=2, col=4):
        self.row = row
        self.col = col
        self.idx = 1

    def getax(self):
        ax = plt.subplot(self.row, self.col, self.idx)
        self.idx += 1
        return ax

layout = layoutclass(cfg["nrows"], cfg["ncols"])

#
# register  new graph modules
#
#

modulelist = [] # a list of graph modules

for k in cfg["modnames"]:
    name='graph_%s' % k
    m = __import__(name)
    c = getattr(m, name)
    modulelist.append( c(params, layout) )

fig.tight_layout(pad=3.2) # w_pad=1.0, h_pad=2.0

#
#
#

lastdbid=0 # this is used to keep track the DB records
params['ts'] = 0

while True:
    profile_t1 = time.time()

    #print 'iterate coolr display loop: t1='+str(profile_t1)

    if fakemode:
        j = fakedata.queryfakedataj()
    else:
        for k in querycmds:
            cmd = cfg[k]
            #print 'iterate through querycmds'
            if cmd == 'dbquerycmd':
                if lastdbid > 0:
                    j = querydataj("%s --gtidx=%d" % (cmd, lastdbid))
                else:
                    j = querydataj(cmd)
                if len(j) > 0:
                    lastdbid = int(j[-1]['dbid'])
            else:
                j = querydataj(cmd)

    profile_t2 = time.time()
    #print 'iterate coolr display loop: t2='+str(profile_t2)+' j='+str(j)
    #iterator = 0

    #while True:

    if not j:
        continue
        print 'not j: continue'

    for e in j:
       	#iterator += 1
      	print >>logf, json.dumps(e)
       	#print 'check key'
       	if not (e.has_key('node') and\
              	e.has_key('sample') and\
               	e.has_key('time') ):
	    print 'Ignore this invalid sample:', json.dumps(e)
            continue

        #print 'set timestamp'
        if params['ts'] == 0:
        	 params['ts'] = e['time']
            	 t = 0
        
        #if iterator%2 == 0:
        
        #print 'iterate coolr display loop: before updating - iter='+str(iter)
        for m in modulelist:
      		m.update(params,e)
        	#print 'module: '+str(m)

    #print 'iterate coolr display loop: before drawing'
    plt.draw()

    profile_t3 = time.time()
    #print 'iterate coolr display loop: t3='+str(profile_t3)

    pausesec = 0.0
    if intervalsec > profile_t3-profile_t1:
        pausesec = intervalsec - (profile_t3-profile_t1)
    if pausesec > 0.0:
        plt.pause(pausesec)

    print 'Profile Time [S]: %.2lf (%.2lf+%.2lf+%.2lf) / Queried %3d items from DB' %\
        (profile_t3-profile_t1+pausesec, profile_t2-profile_t1,\
         profile_t3-profile_t2, pausesec, len(j))

sys.exit(0)
