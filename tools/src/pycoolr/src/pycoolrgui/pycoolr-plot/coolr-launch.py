#!/usr/bin/env python

import sys, os, re
import json
import time
import getopt

from listrotate import *
from clr_utils import *
from layout import *

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


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from clr_matplot_graphs import *
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beacon_subscribe import *


if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 8})


'''
class layoutclass:
    def __init__(self, row=2, col=4):
        self.row = row
        self.col = col
        self.idx = 1

        #rax = plt.axes([0.02, 0.8, 0.08, 0.1])
        #check = CheckButtons(rax, ('Memory', 'MPI_T_PVAR', 'NODE_POWER'), (False, True, False))

        #check.on_clicked(handlerCB)

    def getax(self):
        ax = plt.subplot(self.row, self.col, self.idx)
        self.idx += 1
        return ax

    def distrplots(self):
        ax = f.add_subplot(self.row,self.col,self.idx)
        self.idx += 1
        return ax
'''

#f = Figure(figsize=(cfg["figwidth"], cfg["figheight"]), dpi=100)
#f = Figure(figsize=(15, 8), dpi=100)

def mainLoop():
  print("hello")
  subSpawn()
  #root.after(2000,mainLoop)

ngraphs = len(params['cfg']['appsamples'])
print 'samples ', params['cfg']['appsamples']
data_lr = [listrotate2D(length=params['lrlen']) for i in range(ngraphs)]

#fig = plt.figure( figsize=(cfg["figwidth"],cfg["figheight"]) )
#fig = plt.figure( figsize=(20,15) )
#c = fig.add_subplot(111)

#canvas = FigureCanvasTkAgg(layout.fig, master=root)
layout = layoutclass(cfg["nrows"], cfg["ncols"])

#root = Tk.Tk()
root = layout.root
root.wm_title("COOLR Beacon")

canvas = layout.canvas


#idx = 1

ax = [layout.distrplots(layout.fig) for i in range(ngraphs)]
ytop = [1 for i in range(ngraphs)]
ybot = [1 for i in range(ngraphs)]

#t = arange(0.0, 3.0, 0.01)
#s = sin(2*pi*t)

#a.plot(t, s)
#b.plot(t, s)
#a.set_title('Tk embedding')
#a.set_xlabel('X axis label')
#a.set_ylabel('Y label')
#fig.canvas.set_window_title('COOLR Beacon')

# a tk.DrawingArea

canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

button = Tk.Button(master=root, text='Quit', command=sys.exit)
button.pack(side=Tk.BOTTOM)

var = Tk.IntVar()

c = Tk.Checkbutton(master=root, text='Expand', command=var)
c.pack(side=Tk.BOTTOM)

root.after(2000,mainLoop)
root.mainloop()

