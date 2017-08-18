#!/usr/bin/env python

import sys, os, re, thread
import json
import getopt
import time
import ctypes
import threading
from ctypes import cdll
from ctypes.util import find_library
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from clr_matplot_graphs import *
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from listrotate import *
from clr_utils import *
from layout import *

matplotlib.rcParams.update({'font.size': 6})

# default values

cfgfn = ''
fakemode = False
appcfgfn = ''
targetnode = ''
enclaves = []
#intervalsec = 1.0
intervalsec = 0.5

#matplotlib.rcParams.update({'font.size': 8})

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


if sys.version_info[0] < 3:
    import Tkinter
else:
    import tkinter 

class Coolrsub:

  def __init__(self, master, row=2, col=3):

        # Create a container
        self.frame = Tkinter.Frame(master,width=200,height=100)
        # Create 2 buttons
        #self.button_left = Tkinter.Button(frame,text="< Decrease Slope",
        #                               command=self.decrease)

        self.lock = threading.Lock()

        self.nbsamples = params['cfg']['nbsamples']
        self.listtitles = params['cfg']['appsamples']
        self.nbGraphs = params['cfg']['nbgraphs']
        self.listRecordSample = [-1] * self.nbGraphs
        self.listckbuttons = [None] * self.nbsamples
        self.listSamplesGraphs = [-1] * self.nbsamples
        self.listSamplesAllocated = [-1] * self.nbsamples

        self.dictCheckSamples=dict()
        self.dictSingleSample=dict()
        self.dictSamplesGraphs=dict()

        self.listSamples = []
        #listKeys = {'nameSample','checked','plotted'}    
    
        self.listSamplesClicked = [0] * self.nbsamples     
        self.listUsedGraphs = [0] * self.nbGraphs

        for idx in range(params['cfg']['nbgraphs']):
         self.listUsedGraphs.append(0)

        for idx in range(params['cfg']['nbsamples']):
          #self.listSamplesClicked.append(0)
          #for key in listKeys:
            #self.dictSingleSample[key] = "0"
            #self.dictSingleSample['nameSample'] = params['cfg']['appsamples'][idx]

          self.dictCheckSamples.update({params['cfg']['appsamples'][idx]:self.dictSingleSample}) 
          self.listSamples.append(self.dictSingleSample)

         #for y in (self.dictCheckSamples[x]):
         #  print 'values: ', 
     
        #self.nbsamples = int(params['cfg']['nbsamples'])
        #listappsamples = params['cfg']['appsamples']
        listappsamples = [None] * self.nbsamples
        self.titles = ('Memory Footprint (VmRSS - Resident Set Size)', 'Peak Memory Usage (VmHWM - High Water Mark)', 'rapl::RAPL_ENERGY_PKG (CPU Socket Power in Watts)', 'mem_allocated (Current level of allocated memory within the MPI library)', 'num_malloc_calls (Number of MPIT_malloc calls)','num_free_calls (Number of MPIT_free calls)')
        self.listappsamplescheck = []

        for idx in params['cfg']['appsamplescheck']:
         self.listappsamplescheck.append(idx)

        ckbtnidx = 0
        self.cbVars={}

        for index in range(0,self.nbsamples):
          #index = Tkinter.IntVar()
          self.cbVars[index] = Tkinter.IntVar()
          self.listckbuttons[index] = Tkinter.Checkbutton(self.frame, text=self.listappsamplescheck[index], variable=self.cbVars[index], command=self.checkbtnfn)  
          self.listckbuttons[index].pack(side="left")

        #self.button_update = Tkinter.Button(frame,text="Update",
        #                                command=self.updatebtn)

        #self.button_update.pack(side="left")

        #for k in range(6): print 'list Check buttons: ', self.listckbuttons[k]
        #for k in range(self.nbsamples): print 'cVars[%d] = %d ' % (k, self.cbVars[k].get())
        self.row = row
        self.col = col
        self.idx = 1
        self.ngraphs = 6
    
        self.avail_refresh = 0 

        #self.data_lr = [listrotate2D(length=params['lrlen']) for i in range(self.ngraphs)]
        self.data_lr = [listrotate2D(length=params['lrlen']) for i in range(self.nbsamples)]
 
        fig = Figure(figsize=(20,10), dpi=90)
        #ax = fig.add_subplot(111)

        self.ax = [self.subplotter(fig) for i in range(self.ngraphs)]

        #self.ytop = [1 for i in range(self.ngraphs)]
        #self.ybot = [1 for i in range(self.ngraphs)]

        self.ytop = [1 for i in range(self.nbsamples)]
        self.ybot = [1 for i in range(self.nbsamples)]

        #self.subSpawn()

  	self.root = master

        self.canvas = FigureCanvasTkAgg(fig,master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        master.wm_title("COOLR Beacon")
        #frame.pack(fill=X, padx=5, pady=5)
        self.frame.pack()

        self.subSpawn()

  def subplotter(self, fig):       
        ax = fig.add_subplot(self.row, self.col, self.idx)
        ax.dist = 20
        #ax = fig.subplot(self.row, self.col, self.idx)
        #print 'row='+str(self.row)+ ' col='+str(self.col)+' idx='+str(self.idx)
        self.idx += 1
        return ax

  def clearplot(self,idxGraph):

       print 'clearplot: idxGraph=', idxGraph
       ax = self.ax[idxGraph]
       ax.cla()    
       #ax.clf()
       #ax.clear()
       #self.canvas.draw()

  def updateplot(self, gxsec, t, idxGraph, idxSample):

        #print 'updateplot: idxGraph=%d, idxSample=%d' %(idxGraph, idxSample)
        ax = self.ax[idxGraph]
        #pdata = self.data_lr[j]
        #pdata = self.data_lr[idxGraph]
        pdata = self.data_lr[idxSample]

        label = params['cfg']['units'][idxSample]

        ax.cla()
        ax.set_xlim([t-gxsec, t])

        #print 'get x and y'
        x = pdata.getlistx()
        y = pdata.getlisty()

        #print 'get ymax and ymin'
        ymax = pdata.getmaxy()
        ymin = pdata.getminy()
        if ymax > self.ytop[idxGraph]:
          self.ytop[idxGraph] = ymax * 1.1
          #self.ytop[idxGraph] = ymax * 1.5
          #self.ytop[i] = ymax * 1.1

        if self.ybot[idxGraph] == 1 or ymin < self.ybot[idxGraph]:
          self.ybot[idxGraph] = ymin*.9
          #self.ybot[idxGraph] = ymin*1.2
          #self.ybot[i] = ymin*.9

        ax.set_ylim([self.ybot[idxGraph], self.ytop[idxGraph]])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(1,0))

        #print 'ax plot'
        #ax.plot(x, y, label='', color=self.colors[i], lw=1.2)
        #print 'plot x and y'
        ax.plot(x, y, 'rs', lw=2)
        #ax.bar(x, y, width = .6, edgecolor='none', color='#77bb88' )
        #ax.plot(x,y, 'ro', scaley=True, label='')

        #print 'ax set x y label'
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(label)

        #print 'ax set title'
        ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.listtitles[idxSample], params['targetnode']) )


  def compute_data(self,t,sample,mean_val,total_val,num_vals):
        
        t = sample['time'] - params['ts']
        goodrecord=0
        for i in range(self.ngraphs):
                ref1=params['cfg']
                ref2=ref1['appsamples']
                #print 'ref - appsamples:'+str(ref2)
                ref3=ref2[i]
                #print 'ref3 : '+str(ref3)
                #print 'sample list:'+str(sample)
                #print 'check if ref in sample'
                if ref3 in sample:
                        ref4=sample[ref3]

                        total_val=total_val+ref4
                        num_vals=num_vals+1
                        mean_val=total_val/num_vals
                        print 'display record ref4='+str(ref4)
                        self.data_lr[i].add(t,ref4)
                        #self.data_lr[i].add(t,mean_val)
                        goodrecord=1

        if goodrecord==0:
                print 'bad record'
                return        


  def updategui(self, params, sample):
        if self.ngraphs == 0:
            return

        mean_val = 0
        total_val=0
        num_vals=0

	#print 'starting update gui: e=', sample
        if sample['node'] == params['targetnode'] and sample['sample'] == 'tau':
            #
            # data handling
            #
            t = sample['time'] - params['ts']
	    goodrecord=0
            for i in range(self.ngraphs):
            #for i in range(self.nbsamples):
                if self.listRecordSample[i] != -1:
                  j = self.listRecordSample[i]
	          ref1=params['cfg']
  		  ref2=ref1['appsamples']
                  #print 'ref - appsamples:'+str(ref2)
		  #ref3=ref2[i]
		  ref3=ref2[j]
                  #print 'ref3 : '+str(ref3)
                  #print 'sample list:'+str(sample)
                  #print 'check if ref in sample'
		  if ref3 in sample:
			ref4=sample[ref3]

                        #total_val=total_val+ref4
                        #num_vals=num_vals+1
                        #mean_val=total_val/num_vals
                        #print 'display record ref4='+str(ref4)
                	#self.data_lr[i].add(t,ref4)
                	self.data_lr[j].add(t,ref4)
                	#self.data_lr[i].add(t,mean_val)
			goodrecord=1

	    if goodrecord==0:
		#print 'bad record'
		return

            #
            # graph handling
            #
            gxsec = params['gxsec']
            #
            #

            #print 'parse graphs'
            for i in range(self.ngraphs):
              if self.listRecordSample[i] != -1:
                j = self.listRecordSample[i]
       
                #self.lock.acquire()
                #self.avail_refresh = 0
                ax = self.ax[i]
                #pdata = self.data_lr[i]
                pdata = self.data_lr[j]
                #label = params['cfg']['appsamples'][i]
                #label = params['cfg']['units'][i]
                label = params['cfg']['units'][j]
                try:
                  ax.cla()
                except Exception as errCla:
                  print 'update_gui: Error cla(): ', type(errCla), errCla

                ax.set_xlim([t-gxsec, t])
                #print 'get x and y'
                x = pdata.getlistx()
                y = pdata.getlisty()

                #print 'get ymax and ymin'
                ymax = pdata.getmaxy()
                ymin = pdata.getminy()

                #self.avail_refresh = 1
                #if ymax > self.ytop[i]:
                if ymax > self.ytop[j]:
                    self.ytop[j] = ymax * 1.1
                    #self.ytop[i] = ymax * 1.1

		#if self.ybot[i] == 1 or ymin < self.ybot[i]:
		if self.ybot[j] == 1 or ymin < self.ybot[j]:
		    self.ybot[j] = ymin*.9
		    #self.ybot[i] = ymin*.9

                #ax.set_ylim([self.ybot[i], self.ytop[i])
                ax.set_ylim([self.ybot[j], self.ytop[j]])
                ax.ticklabel_format(axis='y', style='sci', scilimits=(1,0))

                #print 'ax plot'
                #ax.plot(x, y, label='', color=self.colors[i], lw=1.2)
                #print 'begin plot'
                ax.plot(x, y, 'rs', lw=1)
                #print 'end plot'
                #ax.bar(x, y, width = .6, edgecolor='none', color='#77bb88' )
                #ax.plot(x,y, 'ro', scaley=True, label='')

                #print 'ax set x y label'
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(label)

                #print 'ax set title'
                #ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )
                ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.listtitles[j], params['targetnode']) )

                #self.lock.release()
            #rax = plt.axes([0.02, 0.4, 0.08, 0.1])
            #check = CheckButtons(rax, ('Memory', 'MPI_T_PVAR', 'NODE_POWER'), (False, True, True))
      
            #check.on_clicked(handlerCB)

        else:
            t = sample['time'] - params['ts']
            gxsec = params['gxsec']

            #for i in range(self.ngraphs):
            #    self.ax[i].set_xlim([t-gxsec, t])
            #    self.ax[i].set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )
	#print 'ending update'

  def subscribe(self,libarbjsonbeep):

     print 'start thread with Subscribe'

     listargs = ['MEMORY','NODE_POWER_WATTS','MPI_T_PVAR']

     #libarbjsonbeep.subscribe(2, "MPI_T_PVAR")
     libarbjsonbeep.subscribe(4, "MEMORY", "NODE_POWER_WATTS","MPI_T_PVAR")


  def refresh_plot_loop():

       #print 'loop readEvents'
       #print 'main loop readEvents'
       profile_t1 = time.time()

       #params['ts'] = 0
       #result = read_ts()
       listTmpEvents = []
       #print 'start readEvents main loop'

       read_nb_events = libarbjsonbeep.read_events_size
       read_nb_events.argtype = ()
       read_nb_events.restype = ctypes.c_int

       resultNbEvents = read_nb_events()
       nb_events = resultNbEvents

       if nb_events != tmp_nb_events:
         update = 1
       else:
         update = 0
  
       #print 'Number of events', nb_events
 
       #events_index = nb_events-1
       if high_index != nb_events-1 and high_index != 0:
         low_index = high_index+1

       high_index = nb_events-1

       #print 'low_index='+str(low_index)+' high_index='+str(high_index)

       profile_t2 = time.time()

       if update == 1:
         read_payload = libarbjsonbeep.read_payload
         read_payload.argtype = (ctypes.c_int)
         read_payload.restype = ctypes.POINTER(ctypes.c_char)
    
         for i in range(low_index, high_index+1):
           #resultPayload = read_payload(1)
           #resultPayload = read_payload(events_index)
           resultPayload = read_payload(i)

           profile_t3 = time.time()
           #print 'main loop sub: start building payload' 
       
           payload = ''
           for j in range(0,512):

             if resultPayload[j] == "}":
              payload += resultPayload[j]
              break

             payload += resultPayload[j]


           payload.strip()
           print 'payload =',payload
           try:
             j = json.loads(payload)
           except ValueError as e:
             print 'Failed to load json data: %s' %e
             continue
             #return False

           #listEvents.append(payload)
           listEvents.append(j)
           listTmpEvents.append(j)
           #print 'main loop sub: start building payload' 
           profile_t4 = time.time()
           #for i in range(0,10):
           #print 'stored payload:', payload
 
       tmp_nb_events = nb_events
       #low_index = nb_events

       #time.sleep(2)

       #print 'Test listEvents: ', listEvents
       #if not listEvents:
       #if not listTmpEvents:
         #continue
      #print 'no listEvents'

       #for e in listEvents:
       for e in listTmpEvents:
         #iterator += 1
         #print 'listEvents element ', iterator
         #print >>logf, json.dumps(e)
         #print 'check key'
         #if not (e.has_key('node') and\
         #        e.has_key('sample') and\
         #        e.has_key('time') ):
         if 'node' not in e and\
            'sample' not in e and\
            'time' not in e:
             print 'Ignore this invalid sample:', json.dumps(e)
             continue

         #print 'set timestamp'
         #print 'event element', e
         #print 'event time', e['time']
         if params['ts'] == 0:
               params['ts'] = int(e['time'])
               t = 0

         #print 'iterate coolr display loop: before updating'
         #for m in modulelist:
          #       m.update(params,e)
                #print 'module: '+str(m)
         #print 'updategui'
         profile_t5 = time.time()         
         
         #if iterator % 2 == 0:
         self.updategui(params,e)

         profile_t6 = time.time()
         #if iterator%10 == 0:
         #  print 'draw canvas'
         #  self.canvas.draw()

       #print 'finished parsing listEvents'
       #draw to refresh plotting
       #layout.canvas.draw()
       print 'draw canvas'
       try:
         self.canvas.draw()
       except Exception as errDraw:
         print 'Error drawing canvas: ', type(errDraw), errDraw
       #plt.draw()

       profile_t7 = time.time()
       #print 'iterate coolr display loop: t3='+str(profile_t3)

       pausesec = 0.0
       if intervalsec > profile_t3-profile_t1:
         pausesec = intervalsec - (profile_t3-profile_t1)
       #if pausesec > 0.0:
         #print 'pausesec=%d' %(pausesec)
         #plt.pause(pausesec)
         #plt.pause(3.0)

       #print 'Profile Time [s]: t2=%.2lf, t3=%.2lf, time building json chain = %.2lf, time update gui = %.2lf' %(profile_t2-profile_t1, profile_t3-profile_t1, profile_t5-profile_t4, profile_t7-profile_t6)
       #print 'Profile Time [S]: %.2lf (%.2lf+%.2lf+%.2lf) / Queried %3d items from DB' %\
       # (profile_t3-profile_t1+pausesec, profile_t2-profile_t1,\
       #  profile_t3-profile_t2, pausesec, len(j))


  def readEvents(self,libarbjsonbeep):

    print 'start thread: readEvents'

    low_index = 0
    high_index = 0
    tmp_nb_events = 0
    update = 0
    listEvents = []
    iterator = 0
    params['ts'] = 0

    while True:
      #print 'loop readEvents'
       #print 'main loop readEvents'
       profile_t1 = time.time()
       #self.avail_refresh = 1
       #params['ts'] = 0
       #result = read_ts()
       listTmpEvents = []
       #print 'start readEvents main loop'

       read_nb_events = libarbjsonbeep.read_events_size
       read_nb_events.argtype = ()
       read_nb_events.restype = ctypes.c_int

       resultNbEvents = read_nb_events()
       nb_events = resultNbEvents

       if nb_events != tmp_nb_events:
         update = 1
       else:
         update = 0
  
       #print 'Number of events', nb_events
 
       #events_index = nb_events-1
       if high_index != nb_events-1 and high_index != 0:
         low_index = high_index+1

       high_index = nb_events-1

       #print 'low_index='+str(low_index)+' high_index='+str(high_index)

       profile_t2 = time.time()

       if update == 1:
         read_payload = libarbjsonbeep.read_payload
         read_payload.argtype = (ctypes.c_int)
         read_payload.restype = ctypes.POINTER(ctypes.c_char)
    
         for i in range(low_index, high_index+1):
           #resultPayload = read_payload(1)
           #resultPayload = read_payload(events_index)
           resultPayload = read_payload(i)

           profile_t3 = time.time()
           #print 'main loop sub: start building payload' 
       
           payload = ''
           for j in range(0,512):

             if resultPayload[j] == "}":
              payload += resultPayload[j]
              break

             payload += resultPayload[j]


           payload.strip()
           #print 'payload =',payload
           try:
             j = json.loads(payload)
           except ValueError as e:
             print 'Failed to load json data: %s' %e
             continue
             #return False

           #listEvents.append(payload)
           #listEvents.append(j)
           listTmpEvents.append(j)
           #print 'main loop sub: start building payload' 
           profile_t4 = time.time()
           #for i in range(0,10):
           #print 'stored payload:', payload
 
       tmp_nb_events = nb_events
       #low_index = nb_events

       #time.sleep(2)

       #print 'Test listEvents: ', listEvents
       #if not listEvents:
       if not listTmpEvents:
         continue
      #print 'no listEvents'

       #for e in listEvents:
       for e in listTmpEvents:
         #iterator += 1
         #print 'listEvents element ', iterator
         #print >>logf, json.dumps(e)
         #print 'check key'
         #if not (e.has_key('node') and\
         #        e.has_key('sample') and\
         #        e.has_key('time') ):
         if 'node' not in e and\
            'sample' not in e and\
            'time' not in e:
             print 'Ignore this invalid sample:', json.dumps(e)
             continue

         #print 'set timestamp'
         #print 'event element', e
         #print 'event time', e['time']
         if params['ts'] == 0:
               params['ts'] = int(e['time'])
               t = 0

         #print 'iterate coolr display loop: before updating'
         #for m in modulelist:
          #       m.update(params,e)
                #print 'module: '+str(m)
         #print 'updategui'
         profile_t5 = time.time()         
         
         #if iterator % 2 == 0:
         #self.avail_refresh = 0
         self.lock.acquire()
         self.updategui(params,e)
         self.lock.release()
         #self.avail_refresh = 1
         profile_t6 = time.time()
         #if iterator%10 == 0:
         #  print 'draw canvas'
         #  self.canvas.draw()

       #print 'finished parsing listEvents'
       #draw to refresh plotting
       #layout.canvas.draw()
       #print 'draw canvas'
       #try:
       #  self.canvas.draw() # FIXME: out of stack space (infinite loop ?  Godzilla)
       #except Exception as errDraw:
       #  print 'Error drawing canvas: ', type(errDraw), errDraw
       #plt.draw()

       profile_t7 = time.time()
       #print 'iterate coolr display loop: t3='+str(profile_t3)

       pausesec = 0.0
       if intervalsec > profile_t3-profile_t1:
         pausesec = intervalsec - (profile_t3-profile_t1)
       #if pausesec > 0.0:
         #print 'pausesec=%d' %(pausesec)
         #plt.pause(pausesec)
         #plt.pause(3.0)

       #print 'Profile Time [s]: t2=%.2lf, t3=%.2lf, time building json chain = %.2lf, time update gui = %.2lf' %(profile_t2-profile_t1, profile_t3-profile_t1, profile_t5-profile_t4, profile_t7-profile_t6)
       #print 'Profile Time [S]: %.2lf (%.2lf+%.2lf+%.2lf) / Queried %3d items from DB' %\
       # (profile_t3-profile_t1+pausesec, profile_t2-profile_t1,\
       #  profile_t3-profile_t2, pausesec, len(j))

  def subSpawn(self):

     print 'load beacon subscriber library'
     #libarbjsonbeep = cdll.LoadLibrary(os.getcwd()+'/libarbitraryjsonbeep.so')
     #libarbjsonbeep = cdll.LoadLibrary(os.getcwd()+'/libarbitraryjsonbeepmulsub.so')
     libarbjsonbeep = cdll.LoadLibrary(params['cfg']['libpath']+'/libarbitraryjsonbeepmulsub.so')
     #libmonitor = ctypes.cdll.LoadLibrary(find_library('beaconmonitor'))

     try:
       thread.start_new_thread(self.subscribe,(libarbjsonbeep,))
       thread.start_new_thread(self.readEvents,(libarbjsonbeep,))
       #thread.start_new_thread(self.readEvents,(libarbjsonbeep,))
     except Exception as errThread:
       print "Error: unable to start thread: ", errThread
     
     #while 1:
     #  pass

     
     self.refresh_plot()
     #self.readEvents(libarbjsonbeep)

     #while True:
     #  time.sleep(1)
     #  print 'draw canvas'
     #  try:
     #    self.canvas.draw()
     #  except Exception as errDraw:
     #    print 'Error drawing canvas: ', type(errDraw), errDraw
      

  def parseSamples(self):
      for i in range(self.nbsamples):
        if self.listSamplesAllocated[i] == -1 and self.cbVars[i].get() == 1:
         
          for j in range(self.ngraphs):  
            if self.listUsedGraphs[j] == 0:
              self.listUsedGraphs[j] = 1
              self.listRecordSample[j] = i
              self.listSamplesAllocated[i] = j
              self.listSamplesGraphs[i] = j
            

  def checkbtnfn(self):
       for i in range(self.nbsamples):
         #print 'test check button: idx=%d value=%d click state = %d' %(i, self.cbVars[i].get(), self.listSamplesClicked[i])

 #print'updategui - parse sample index %d' %(i)
         # Check if checkbox related to the current sample has just been checked
         if self.cbVars[i].get() == 1 and self.listSamplesClicked[i] == 0:
            
             #print 'sample %d just selected' %(i) 
             # Mark the check box associated to the current sample as clicked             
             self.listSamplesClicked[i] = 1
             # Parse all graphs
             for j in range(self.ngraphs):

                  #print 'check graph %d for sample %d' %(j, i) 
                  # Check if current graph is not already used
                  #if self.listUsedGraphs[j] == 0 or self.listSamplesAllocated[i] == j:
                  if self.listUsedGraphs[j] == 0:

                    #print 'graph %d for sample %d not used or dedicated: plot' %(j,i) 
                    # Mark current graph as used 
                    self.listUsedGraphs[j] = 1
                    # Record the current graph as plotting the current sample
                    #print 'Record Sample %d for graph %d' %(i,j)
                    self.listRecordSample[j] = i
                    # Mark current sample as allocated to the current graph
                    if self.listSamplesAllocated[i] == -1:
                      self.listSamplesAllocated[i] = j

                    # Associate current graph to the current sample
                    self.listSamplesGraphs[i] = j
                    # Consider next sample 
                    break

         #  Check box has just been clicked off
         if self.cbVars[i].get() == 0 and self.listSamplesClicked[i] == 1:

            #print 'sample %d just unselected: clear plot' %(i)
            # Find graph related to sample associated to unclicked check box
            idxGraph = self.listSamplesGraphs[i]
            self.listSamplesGraphs[i] = -1
            # Clear the record
            graphRecord = self.listSamplesAllocated[i]
            self.listRecordSample[graphRecord] = -1
            self.listSamplesAllocated[i] = -1
            # Clear graph related to the sample
            self.clearplot(idxGraph)
            # The graph is now available
            self.listUsedGraphs[idxGraph] = 0
            # Mark the check box associated to the current sample as unclicked  
            self.listSamplesClicked[i] = 0

            # Reallocate samples
            self.parseSamples()
           
         #if self.cbVars[i].get() == 0:

           #print 'Checkbtnfn: sampled %d unclicked' %(i)   
  
       #if self.


  def refresh_plot(self):
       #print 'refresh_plot - avail: ', self.avail_refresh
       #if self.avail_refresh == 1:
       self.lock.acquire()
       try:
         self.canvas.draw()
         #self.frame.update()
       except Exception as errDraw:
         print 'refresh_plot: Error drawing canvas: ', type(errDraw), errDraw
       self.lock.release()

       self.root.after(1000,self.refresh_plot)      

  def updatebtn(self):
       print 'update buttonupdate button'
       try:
         self.canvas.draw()
       except Exception as errDraw:
         print 'Error drawing canvas: ', type(errDraw), errDraw

  def checkfn(self, idx, text): 
       print 'checkfn'
       print 'Check index=%d text=%s' % (idx,text)
       #print 'Size of listbtnchecked[]= ', len(self.listbtnchecked)
       #self.listbtnchecked[idx] = 1
       

root = Tkinter.Tk()
app = Coolrsub(root,2,3)
root.mainloop()
