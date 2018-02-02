#!/usr/bin/env python

import sys, os, re, thread, signal
from cStringIO import StringIO
import subprocess
import multiprocessing
import json
import sqlite3
import getopt
import time
import ctypes
import threading
import itertools
from ctypes import cdll
from ctypes.util import find_library
import random
import pylab as pl
import matplotlib
import matplotlib.pyplot as plt
import re
#import matplotlib.animation as manimation
#from clr_matplot_graphs import *
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from listrotate import *
from clr_utils import *
from layout import *

#matplotlib.rcParams.update({'font.size': 14})
#matplotlib.rcParams.update({'font.family': 'cursive'})
#matplotlib.rcParams.update({'font.family': 'fantasy'})
#matplotlib.rcParams['font.family'] = 'Helvetica'

#font = FontProperties()
#font.set_family('cursive')

font = {'family' : 'serif',
        'weight' : 'light',
        'size'   : 8}

matplotlib.rc('font', **font)

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
    #cfgfn='configs/demososcerberus.cfg'

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
    targetnode = os.environ['PYCOOLR_NODE']
    #targetnode = cfg['masternode']
if len(enclaves) == 0:
    if cfg.has_key('enclaves'):
        enclaves = cfg['enclaves']

#print 'masternode:', cfg['masternode']
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

#
#
#
try:
    logf = open(cfg["outputfn"], 'w', 0) # unbuffered write
except:
    print 'unable to open', cfg["outputfn"]


#if not fakemode:
#    querycmds = cfg['querycmds']


lrlen=200  # to option
gxsec=120 # graph x-axis sec

#info = querydataj(cfg['queryinfocmd'])[0]

params = {}  # graph params XXX: extend for multinode
params['cfg'] = cfg
#params['info'] = info
params['lrlen'] = lrlen
params['gxsec'] = gxsec
params['cur'] = 0  # this will be updated
params['pkgcolors'] = [ 'blue', 'green' ] # for now
params['targetnode'] = targetnode
params['enclaves'] = enclaves

if sys.version_info[0] < 3:
    import Tkinter
    #from Tkinter import *
    import tkFileDialog
    import tkFont
    from tkFont import Font
    #from Tkinter.FileDialog import askopenfilename
else:
    import tkinter
    from tkinter.font import Font
    #from tkinter.filedialog import askopenfilename


class Coolrsub:

  def __init__(self, master, row=2, col=3):

        # Create a container
        self.frame = Tkinter.Frame(master,width=200,height=100)
        # Create 2 buttons
        #self.button_left = Tkinter.Button(frame,text="< Decrease Slope",
        #                               command=self.decrease)

        self.pubpid = -1
        self.subpid = -1

        self.lock = threading.Lock()

        params = {}
        params['cfg'] = cfg
        self.tool = params['cfg']['tool']
        self.nbsamples = params['cfg']['nbsamples']
        #self.nbcvars = params['cfg']['nbcvars']
        self.listmetrics = params['cfg']['metrics']
        #self.listsamples = params['cfg']['appsamples']
        self.nbGraphs = params['cfg']['nbgraphs']
      
        if self.tool == "sos":
          self.groupcolumns = params['cfg']['groupcolumns']
          self.ranks2 = params['cfg']['ranks2']
          self.sosdbfile = params['cfg']['dbfile']
          self.sos_bin_path = ""
          self.res_sql = [None]
          self.res_min_ts_sql = [None]

        if self.tool == "beacon":
          self.nbcvars = params['cfg']['nbcvars']
          self.listcvars = params['cfg']['cvars']
          self.listcvarsvalues = []
          self.listcfgcvarsarray = params['cfg']['cvarsarray']
          self.listlabelcvarsmetric = []
          self.listcvarsentry = []
          self.listlabelcvarsarrayindex = []
          self.listcvarsarrayindexentry = []
          self.btncvarsupdate = None

        self.metrics = params['cfg']['metrics']
        #self.ranks = params['cfg']['ranks']
        self.ranks = [None] * self.nbsamples
        self.procs = [None] * self.nbsamples
        self.nodes = [None] * self.nbsamples
        self.noderanks = [None] * self.nbsamples
        self.listRecordSample = [-1] * self.nbGraphs
        self.listckbuttons = [None] * self.nbsamples
        self.listSamplesGraphs = [-1] * self.nbsamples
        self.listSamplesAllocated = [-1] * self.nbsamples

        # Connection to the database
        self.conn = None
        #self.ranks = -1
        #self.nodes = -1
        self.group_column = ""

        self.dictCheckSamples=dict()
        self.dictSingleSample=dict()
        self.dictSamplesGraphs=dict()

        #self.rows = ""
        self.rows = [None] * self.nbsamples
        #self.dictcvars = dict(zip(self.listcvars, self.listcvarsvalues))
        #self.dictcvars = dict()

        #listKey = {"a","b","c","d"}
        #dictKey = dict([(key, []) for key in listKey])
        #print "dictKey: ", dictKey

        #self.dictcvars.fromkeys(self.listcvars)
 
        #print "list cvars: ", self.listcvars 
        #print "dict cvars: ", self.dictcvars
        #for key in self.dictcvars.items():
        #  print "dictcvars: ",key
        #print 'dictionary cvars: ',self.d ictcvars

        self.listSamples = []
   
        #listKeys = {'nameSample','checked','plotted'}    
       
        self.listSamplesClicked = [0] * self.nbsamples     
        self.listUsedGraphs = [-1] * self.nbGraphs

        #self.listFontPolicies = ['Arial', 'Times', 'Helvetica', 'Liberation Sans', 'Liberation Serif']
        self.listFontSizes = [6,8,10,12,14,16,18,20,24,28,32,36]
        self.listFontFamily = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
        self.listFontWeight = ['ultralight','light','normal','regular','book','medium','roman','semibold','demibold','demi','bold','heavy','extra bold','black']

        self.list_fonts = list( tkFont.families() )

        self.selectedFontPolicy = None
        self.selectedFontSize = None
        self.selectedFontWeight = None
       
        # create a custom font
        self.customFont = tkFont.Font(family="Helvetica", size=12)
 
        for idx in range(params['cfg']['nbgraphs']):
         self.listUsedGraphs.append(-1)

        #for idx in range(params['cfg']['nbsamples']):
          #self.listSamplesClicked.append(0)
          #for key in listKeys:
            #self.dictSingleSample[key] = "0"
            #self.dictSingleSample['nameSample'] = params['cfg']['appsamples'][idx]

          #self.dictCheckSamples.update({params['cfg']['appsamples'][idx]:self.dictSingleSample}) 
          #self.dictCheckSamples.update({params['cfg']['metrics'][idx]:self.dictSingleSample}) 
          #self.listSamples.append(self.dictSingleSample)

         #for y in (self.dictCheckSamples[x]):
         #  print 'values: ', 
     
        #self.nbsamples = int(params['cfg']['nbsamples'])
        #listappsamples = params['cfg']['appsamples']
        listappsamples = [None] * self.nbsamples
        #self.titles = ('Memory Footprint (VmRSS - Resident Set Size)', 'Peak Memory Usage (VmHWM - High Water Mark)', 'rapl::RAPL_ENERGY_PKG (CPU Socket Power in Watts)', 'mem_allocated (Current level of allocated memory within the MPI library)', 'num_malloc_calls (Number of MPIT_malloc calls)','num_free_calls (Number of MPIT_free calls)')

        ckbtnidx = 0
        self.cbVars={}

        #for index in range(0,self.nbsamples):
          #index = Tkinter.IntVar()
          #self.cbVars[index] = Tkinter.IntVar()
          #self.listckbuttons[index] = Tkinter.Checkbutton(self.frame, text=self.listappsamplescheck[index], variable=self.cbVars[index], command=self.checkbtnfn)  
          #self.listckbuttons[index].pack(side="left")

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
        #self.data_lr = [listrotate2D(length=params['lrlen']) for i in range(self.nbsamples)]
        self.data_lr = [listrotate2D(length=200) for i in range(self.nbsamples)]
 
        fig = Figure(figsize=(35,18), dpi=80)
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
        #self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)
        master.wm_title("COOLR Beacon")
        #frame.pack(fill=X, padx=5, pady=5)
        self.frame.pack()

        menubar = Tk.Menu(root)
        filemenu = Tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Preferences", menu=filemenu)

        filemenu.add_command(label="Metrics", command=self.metricsmenu)
  
        if self.tool == "beacon":
          filemenu.add_command(label="CVARS", command=self.cvarsmenu)

        filemenu.add_command(label="Fonts", command=self.fontmenu)
        filemenu.add_command(label="Exit", command=self.exitPycoolr)

        try:       
          root.config(menu=menubar)
        except AttributeError as attErr:
          print 'menu Exception: ', type(attErr), attErr 

        #self.winPvars()
        #self.winCvars()
  
        self.subSpawn()

  def try_execute(self, c, statement, parameters=None):
    try:
        if parameters:
            c.execute(statement,parameters);
        else:
            c.execute(statement);
    except sqlite3.Error as e:
        print("database error...", e.args[0])

  def open_connection(self):
    global conn
    # check for file to exist
    #print ("Checking for file: ", sqlite_file)
    print "Checking for file: ", self.sosdbfile
    while not os.path.exists(self.sosdbfile):
        print "Waiting on file: ", sqlite_file
        time.sleep(1)

    #print("Connecting to: ", sqlite_file)
    print "Connecting to: ", self.sosdbfile
    # Connecting to the database file
    #conn = sqlite3.connect(sqlite_file)
    #fd = os.open(sqlite_file, os.O_RDONLY)
    #conn = sqlite3.connect('/dev/fd/%d' % fd)
    #url = 'file:' + sqlite_file + '?mode=ro'
    #url = 'file:' + sqlite_file
    #conn = sqlite3.connect(url, uri=True)
    #conn = sqlite3.connect(sqlite_file)
    conn = sqlite3.connect(self.sosdbfile)
    conn.isolation_level=None
    c = conn.cursor()
    #c.execute('PRAGMA journal_mode=WAL;')
    #c.execute('PRAGMA synchronous   = ON;')
    #c.execute('PRAGMA cache_size    = 31250;')
    #c.execute('PRAGMA cache_spill   = FALSE;')
    #c.execute('PRAGMA temp_store    = MEMORY;')
    return c

  def btnfontsupdate(self):
       print 'Update font'
       if self.selectedFontPolicy or self.selectedFontSize or self.selectedFontWeight:
         matplotlib.rcParams.update({'font.size': self.selectedFontSize, 'font.family': self.selectedFontPolicy})
         #self.customFont.configure(family=self.selectedFontPolicy)
         #self.customFont.configure(size=self.selectedFontSize)
         font = {'family' : self.selectedFontPolicy, 'weight': self.selectedFontWeight, 'size': self.selectedFontSize}
         matplotlib.rc('font', **font)

  def ckbtnFontBold(self):
        print 'Bold selected'

  def ckbtnFontItalic(self):
        print 'Italic selected'

  def ckbtnFontUnderline(self):
        print 'Underline selected'
      
  def browsefontpolicy(self):
        print 'browsefontpolicy'
        fontpolicydiag = tkFileDialog.askopenfilename(filetypes=[("Text files","*.fft")])

  def onselectFontPolicy(self,evt):
        w = evt.widget
        selection = w.curselection()
        value = w.get(selection[0])
        self.selectedFontPolicy = value
        print 'select font: ', value

  def onselectFontSize(self, evt):
       print 'select font size'  
       w = evt.widget
       selection = w.curselection()
       value = w.get(selection[0])
       self.selectedFontSize = value
       print 'select font: ', value

  def onselectFontWeight(self, evt):
       print 'select font weight'  
       w = evt.widget
       selection = w.curselection()
       value = w.get(selection[0])
       self.selectedFontWeight = value
       print 'select font: ', value

  def loadFontPolicy(self):
        fontpolicydiag = askopenfilename(filetypes=(("*.fft"))) 

  #def perfeventsmenu(self):

  #def cvarsmenu(self):

  def fontmenu(self):
        print 'nothing'

        self.paramswin = Tk.Tk()
        self.paramswin.title("Fonts: family, size and weight")
 
        #w = 290
        #h = 150
        #RWidth = self.paramswin.winfo_screenwidth()
        #RHeight = self.paramswin.winfo_screenheight()
        #x = (RWidth - w)/2
        #y = (RHeight - h)/2
        #self.pvarswin.geometry("%dx%d" % (RWidth,RHeight))
        #pvarswin.geometry('%dx%d+%d+%d' % (w,h,x,y)) 
        #f1 = Tk.Frame(pvarswin) 
        #f1 = Tk.Frame(pvarswin,width=150,height=100) 
        #fparams = Tk.Frame(self.paramswin, width=200, height=100)
        self.fparams = Tk.Frame(self.paramswin)
        self.fparams.grid(row=0,column=0, sticky="NS")
        
        #self.labelfontfamily=Tk.Label(self.paramswin, text="Font Family")
        #self.labelfontfamily.grid(row=0,column=0,sticky='ne')

        self.sparamsfontpolicies = Tk.Scrollbar(self.fparams)
        self.lbFontPolicies = Tk.Listbox(self.fparams,exportselection=0,width=20,height=10)
        self.lbFontPolicies.grid(row=1,column=0,rowspan=4,sticky="ne")
        for i in range(len(self.listFontFamily)): self.lbFontPolicies.insert(i, self.listFontFamily[i])
        self.sparamsfontpolicies.config(command = self.lbFontPolicies.yview)
        self.lbFontPolicies.config(yscrollcommand = self.sparamsfontpolicies.set)
        self.lbFontPolicies.bind('<<ListboxSelect>>', self.onselectFontPolicy)

        #self.labelfontsize=Tk.Label(self.paramswin, text="Font Size")
        #self.labelfontsize.grid(row=0,column=3,sticky='ne')

        self.sparamsfontsizes = Tk.Scrollbar(self.fparams)
        self.lbFontSizes = Tk.Listbox(self.fparams,exportselection=0,width=20,height=10)
        self.lbFontSizes.grid(row=1,column=3,rowspan=3,sticky="ne")
        #l1 = Tk.Listbox(f1,selectmode='multiple',width=80,height=40)
        for i in range(len(self.listFontSizes)): self.lbFontSizes.insert(i, self.listFontSizes[i])
        self.sparamsfontsizes.config(command = self.lbFontSizes.yview)
        self.lbFontSizes.config(yscrollcommand = self.sparamsfontsizes.set)
        self.lbFontSizes.bind('<<ListboxSelect>>', self.onselectFontSize)

        self.sparamsfontweight = Tk.Scrollbar(self.fparams)
        self.lbFontWeight = Tk.Listbox(self.fparams,exportselection=0,width=20,height=10)
        self.lbFontWeight.grid(row=1,column=4,rowspan=3,sticky="ne")
        #l1 = Tk.Listbox(f1,selectmode='multiple',width=80,height=40)
        for i in range(len(self.listFontWeight)): self.lbFontWeight.insert(i, self.listFontWeight[i])
        self.sparamsfontweight.config(command = self.lbFontWeight.yview)
        self.lbFontWeight.config(yscrollcommand = self.sparamsfontweight.set)
        self.lbFontWeight.bind('<<ListboxSelect>>', self.onselectFontWeight)

        self.updateFontButton= Tk.Button(self.paramswin, text="Update", command=self.btnfontsupdate)
        self.updateFontButton.grid(row=4, column=4, sticky="we")

        #lbfonts.pack(side = Tk.LEFT, fill = Tk.Y)
        #sparams.pack(side = Tk.RIGHT, fill = Tk.Y)
        #fparams.pack()
 
  #def winPvars(self):
  def metricsmenu(self):
    	# this is the child window
        self.pvarswin = Tk.Tk()
        self.pvarswin.title("metrics") 
        #pvarswin = Tk.Tk()
        #pvarswin.title("PVARS") 
        #RWidth = 150
        #RHeight = 100
        w = 290
        h = 150
        RWidth = self.pvarswin.winfo_screenwidth()
        RHeight = self.pvarswin.winfo_screenheight()
        x = (RWidth - w)/2
        y = (RHeight - h)/2

        self.f1 = Tk.Frame(self.pvarswin, width=200, height=100) 
	self.l1 = Tk.Listbox(self.f1,selectmode='multiple',width=100,height=40)

        #self.pvarswin.geometry("%dx%d" % (RWidth,RHeight))
        #pvarswin.geometry('%dx%d+%d+%d' % (w,h,x,y)) 
        #f1 = Tk.Frame(pvarswin) 
        #f1 = Tk.Frame(pvarswin,width=150,height=100) 
	s1 = Tk.Scrollbar(self.f1) 
	#l1 = Tk.Listbox(f1,selectmode='multiple',width=80,height=40)
	for i in range(self.nbsamples): self.l1.insert(i, self.listmetrics[i]) 
	s1.config(command = self.l1.yview) 
	self.l1.config(yscrollcommand = s1.set) 
        self.l1.bind('<<ListboxSelect>>', self.onselectmetrics)
	self.l1.pack(side = Tk.LEFT, fill = Tk.Y) 
	s1.pack(side = Tk.RIGHT, fill = Tk.Y) 
	self.f1.pack()

  def cvarsmenu(self):
        # this is the child window
        self.cvarswin = Tk.Tk()
        self.cvarswin.title("CVARS")

        self.cvarsmetricvar=Tk.StringVar()
        self.cvarsmetricvar.set('CVAR METRIC')

        self.cvarsindexrow = 3

        RWidth = self.cvarswin.winfo_screenwidth()
        RHeight = self.cvarswin.winfo_screenheight()

        self.f2 = Tk.Frame(self.cvarswin)
        self.f2.grid(row=0,column=0, sticky="n")

        s2 = Tk.Scrollbar(self.f2)
        self.l2 = Tk.Listbox(self.f2, selectmode='multiple',width=40,height=20)
        for i in range(self.nbcvars): self.l2.insert(i, self.listcvars[i])
        self.l2.grid(row=0,column=0,rowspan=4,sticky="nwe")
        s2.config(command = self.l2.yview)
        self.l2.config(yscrollcommand = s2.set)
        self.l2.bind('<<ListboxSelect>>', self.onselectcvars)

        self.stepCvarsUpdate = Tk.LabelFrame(self.cvarswin, text="CVARS Update")
        self.stepCvarsUpdate.grid(row=0,column=1,rowspan=5,columnspan=4, sticky='NS')
        #self.option=Tk.OptionMenu(f2, stvar, "one", "two", "three")
        #self.labelcvars=Tk.Label(self.f2, text="TAU_MPI_T_CVAR_METRICS:").grid(row=0,column=2, sticky="nw")
        #self.labelcvarscontent=Tk.Label(self.f2, text="").grid(row=0,column=5, sticky="ne")
        #self.labelcvarscontent=Tk.Label(self.stepCvarsUpdate, textvariable=self.cvarsmetricvar)
        self.labelcvarsmetric=Tk.Label(self.stepCvarsUpdate, text="CVAR METRIC")
        self.labelcvarsmetric.grid(row=2,column=3,sticky='NS')
        #self.option.grid(row=0,column=1,sticky="nwe")
        #self.option.grid(row=0,column=1,sticky="nwe")


  def exitPycoolr(self):
       os.kill(self.subpid, signal.SIGTERM)
       os.kill(self.pubpid, signal.SIGTERM)
       self.destroy() 
       #root.quit() 
       #sys.exit()

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
        #ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.listtitles[idxSample], params['targetnode']) )
        ax.set_title('%s: %s' % (params['cfg']['appname'], self.listmetrics[idxSample]) )


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

  def updateguibeacon(self, params, sample):
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
  		  ref2=ref1['metrics']
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
                #ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.listtitles[j], params['targetnode']) )
                ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.listmetrics[j], params['targetnode']) )

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

  def updateguisos_db(self, params, graphidx, recordidx, sample):
        if self.ngraphs == 0:
            return

        mean_val  = 0
        total_val = 0
        num_vals  = 0
        newplot = True
        graphs = [None, None, None, None, None, None]
        axises = [None, None, None, None, None, None]

        #print '[PYCOOLR] Starting update gui'
        #if sample['node'] == params['targetnode'] and sample['sample'] == 'tau':
            #
            # data handling
            #
            #t = sample['time'] - params['ts']
            #
            # graph handling
            #
        gxsec = params['gxsec']
            #
            #

        #print 'parse graphs'
        metric_value = max(sample[1],0)
        numeric = re.search(r'\d+', metric_value)
        metric_value_num = numeric.group()
        metric_value_float = float(metric_value_num)
        metric_value_int = int(metric_value_float)
        pack_time = sample[2] - min_timestamp

        #print 'metric_value: ', metric_value
        #print 'metric_value_float: ', metric_value_float
        #print 'metric_value_int: ', metric_value_int
        #print 'pack_time ', pack_time
        #print 'graphidx: %d, recordidx: %d' %(graphidx,recordidx)
        #print("Making numpy array of: metric_values"

        self.data_lr[recordidx].add(pack_time,metric_value_int)
        #self.lock.acquire()
        #self.avail_refresh = 0
        ax = self.ax[graphidx]
        #pdata = self.data_lr[i]
        pdata = self.data_lr[recordidx]
        #label = params['cfg']['appsamples'][i]
        #label = params['cfg']['units'][i]
        label = params['cfg']['units'][recordidx]
        try:
           ax.cla()
        except Exception as errCla:
          print 'update_gui: Error cla(): ', type(errCla), errCla

        ax.set_xlim([pack_time-gxsec, pack_time])
        #print 'get x and y'
        x = pdata.getlistx()
        y = pdata.getlisty()

        #print 'get ymax and ymin'
        ymax = pdata.getmaxy()
        ymin = pdata.getminy()

        #self.avail_refresh = 1
        #if ymax > self.ytop[i]:
        if ymax > self.ytop[recordidx]:
          self.ytop[recordidx] = ymax * 1.1
          #self.ytop[i] = ymax * 1.1

        #if self.ybot[i] == 1 or ymin < self.ybot[i]:
        if self.ybot[recordidx] == 1 or ymin < self.ybot[recordidx]:
          self.ybot[recordidx] = ymin*.9
                    #self.ybot[i] = ymin*.9

        #ax.set_ylim([self.ybot[i], self.ytop[i])
        ax.set_ylim([self.ybot[recordidx], self.ytop[recordidx]])
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

        #ax.set_title('%s: %s' % (params['cfg']['metrics'], self.metrics[recordidx]))
        ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.metrics[recordidx], params['targetnode']) )

            #for i in range(self.ngraphs):
            #    self.ax[i].set_xlim([t-gxsec, t])
            #    self.ax[i].set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )
        #print 'ending update'


  def updateguisos(self, params, graphidx, recordidx, sample):
        if self.ngraphs == 0:
            return

        mean_val  = 0
        total_val = 0
        num_vals  = 0
        newplot = True
        graphs = [None, None, None, None, None, None]
        axises = [None, None, None, None, None, None]

	#print '[PyCOOLR - SOS] Starting update gui: sample= ', sample
        #if sample['node'] == params['targetnode'] and sample['sample'] == 'tau':
            #
            # data handling
            #
            #t = sample['time'] - params['ts']
            #
            # graph handling
            #
        gxsec = params['gxsec']
            #
            #
        #for i in len(sample):
        # sample[i].replace('"', '')

        #print 'parse graphs'
        #print "before sample: metric=%s, value=%s, ts=%s" %(repr(sample[1]),repr(sample[2]),repr(sample[3]))
        sample_1 = sample[1].replace('\"', '')
        sample_2 = sample[2].replace('\"', '')
        sample_3 = sample[3].replace('\"', '')

        sample_value = float(sample_2)
        #sample_value = float("0.00000000000000000000")
        metric_value = max(sample_value,0)
        #numeric = re.search(r'\d+', metric_value)
        #metric_value_num = numeric.group()
        #metric_value_float = float(metric_value_num)
        #metric_value_int = int(metric_value_float)
        time_stamp = float(sample_3)
        #time_stamp = float("1510105643.06971")
        pack_time = time_stamp - min_timestamp

        #print 'metric_value: ', metric_value
        #print 'metric_value_float: ', metric_value_float
        #print 'metric_value_int: ', metric_value_int
        #print 'pack_time ', pack_time
        #print 'graphidx: %d, recordidx: %d' %(graphidx,recordidx)
        #print("Making numpy array of: metric_values"

        #self.data_lr[recordidx].add(pack_time,metric_value_int)
        self.data_lr[recordidx].add(pack_time,metric_value)
        #self.lock.acquire()
        #self.avail_refresh = 0
        ax = self.ax[graphidx]
        #pdata = self.data_lr[i]
        pdata = self.data_lr[recordidx]
        #label = params['cfg']['appsamples'][i]
        #label = params['cfg']['units'][i]
        label = params['cfg']['units'][recordidx]
        try:
           ax.cla()
        except Exception as errCla:
          print 'update_gui: Error cla(): ', type(errCla), errCla

        ax.set_xlim([pack_time-gxsec, pack_time])
        #print 'get x and y'
        x = pdata.getlistx()
        y = pdata.getlisty()

        #print 'get ymax and ymin'
        ymax = pdata.getmaxy()
        ymin = pdata.getminy()

        #self.avail_refresh = 1
        #if ymax > self.ytop[i]:
        if ymax > self.ytop[recordidx]:
          self.ytop[recordidx] = ymax * 1.1
          #self.ytop[i] = ymax * 1.1

	#if self.ybot[i] == 1 or ymin < self.ybot[i]:
	if self.ybot[recordidx] == 1 or ymin < self.ybot[recordidx]:
          self.ybot[recordidx] = ymin*.9
		    #self.ybot[i] = ymin*.9

        #ax.set_ylim([self.ybot[i], self.ytop[i])
        ax.set_ylim([self.ybot[recordidx], self.ytop[recordidx]])
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

        #ax.set_title('%s: %s' % (params['cfg']['metrics'], self.metrics[recordidx]))
        ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.metrics[recordidx], params['targetnode']) )

            #for i in range(self.ngraphs):
            #    self.ax[i].set_xlim([t-gxsec, t])
            #    self.ax[i].set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )
	#print 'ending update'

 
  def subscribe(self,libarbjsonbeep):

     print 'start thread with Subscribe'

     listargs = ['MEMORY','NODE_POWER_WATTS','MPI_T_PVAR']

     libarbjsonbeep.subscribe(2, "MPI_T_PVAR")
     #libarbjsonbeep.subscribe(4, "MEMORY", "NODE_POWER_WATTS","MPI_T_PVAR")

  def publish(self,libarbpubcvars):

     print 'start thread with Publish'

     #listargs = ['MEMORY','NODE_POWER_WATTS','MPI_T_PVAR']

     #libarbjsonbeep.subscribe(2, "MPI_T_PVAR")
     libarbpubcvars.publish()


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

  def make_index(self,c):
    sql_statement = ("create index foo2 on tblvals(guid);")
    print("make_index Executing query")
    self.try_execute(c,sql_statement);

  def get_ranks(self,c):
    sql_statement = ("select distinct comm_rank, process_id from tblpubs where title not like 'system monitor' and title not like 'process monitor: %' order by comm_rank;")
    print("get_ranks Executing query")
    self.try_execute(c,sql_statement);
    all_rows = c.fetchall()
    ranks = np.array([x[0] for x in all_rows])
    procs = np.array([x[1] for x in all_rows])
    ranklen = len(ranks)
    if ranklen > 10:
        smallranks = [0]
        for i in range(1,4):
            candidate = random.randrange(1, ranklen-1)
            while candidate in smallranks:
                candidate = random.randrange(1, ranklen-1)
            smallranks.append(candidate)
        smallranks.append(int(ranklen-1))
        smallranks2 = []
        smallprocs2 = []
        for index in smallranks:
            smallranks2.append(ranks[index])
            smallprocs2.append(procs[index])
        return np.array(sorted(smallranks2)), np.array(sorted(smallprocs2))
    else:
        return ranks, procs

  def get_nodes(self,c):
    sql_statement = ("select distinct node_id, min(comm_rank) from tblpubs group by node_id order by comm_rank;")
    print("get_nodes Executing query")
    self.try_execute(c,sql_statement);
    all_rows = c.fetchall()
    nodes = np.array([x[0] for x in all_rows])
    ranks = np.array([x[1] for x in all_rows])
    nodelen = len(nodes)
    if nodelen > 10:
        smallnodes = [0]
        for i in range(1,4):
            candidate = random.randrange(1, nodelen-1)
            while candidate in smallnodes:
                candidate = random.randrange(1, nodelen-1)
            smallnodes.append(candidate)
        smallnodes.append(int(nodelen-1))
        smallnodes2 = []
        smallranks2 = []
        for index in sorted(smallnodes):
            smallnodes2.append(nodes[index])
            smallranks2.append(ranks[index])
        return np.array(smallnodes2), np.array(smallranks2)
    else:
        return nodes,ranks

  def get_min_timestamp_db(self,c):
    global min_timestamp
    sql_statement = ("select min(time_pack) from tblvals;")
    print("get_min_timestamp Executing query")
    self.try_execute(c,sql_statement);
    all_rows = c.fetchall()
    ts = np.array([x[0] for x in all_rows])
    min_timestamp = ts[0]
    print("min timestamp: ", min_timestamp)


  def get_min_timestamp(self):
    global min_timestamp
    sql_statement = ("SELECT min(time_pack) FROM viewCombined;")
    print("get_min_timestamp Executing query")
 
    print "sql statement: ", sql_statement
    #self.try_execute(c, sql_statement)
    os.environ['SOS_SQL'] = sql_statement
    sos_bin_path = os.environ.get('SOS_BIN_DIR')
    print 'SOS BIN path: ', sos_bin_path
    os.system('cd '+ sos_bin_path)
    print 'current dir: ', os.getcwd()
    # Redirect stdout of passed command into a string

    soscmd = sos_bin_path + "/demo_app_silent --sql SOS_SQL"
    print 'soscmd: ', soscmd
    tmp_res_min_ts_sql = subprocess.check_output(soscmd, shell=True)

    #self.res_min_ts_sql = tmp_res_min_ts_sql.splitlines()
    print 'get min ts: tmp res sql=', tmp_res_min_ts_sql
    res_min_ts_sql = tmp_res_min_ts_sql.splitlines()
    print "List of result SQL MIN TS: ", res_min_ts_sql
    min_ts_rows = res_min_ts_sql[1].split(",")
    print "List of result SQL MIN TS values: ", min_ts_rows
    # Remove first element of SQL result 
    #ts = np.array([x[0] for x in min_ts_rows])
    str_min_timestamp = min_ts_rows[0].replace('\"', '')
    min_timestamp = float(str_min_timestamp)
    #print("str min_timestamp=%s, min timestamp=%f: ", str_min_timestamp, min_timestamp)


  def req_sql_db(self, c, ranks, ranks2, group_column, metric):
    print 'req_sql entering'
    for r in ranks:
        sql_statement = ("SELECT distinct tbldata.name, tblvals.val, tblvals.time_pack, tblpubs.comm_rank FROM tblvals INNER JOIN tbldata ON tblvals.guid = tbldata.guid INNER JOIN tblpubs ON tblpubs.guid = tbldata.pub_guid WHERE tblvals.guid IN (SELECT guid FROM tbldata WHERE tbldata.name LIKE '" + metric + "') AND tblpubs." + group_column)
        """
        if isinstance(r, int):
            sql_statement = (sql_statement + " = " + str(r) + " order by tblvals.time_pack;")
        else:
            sql_statement = (sql_statement + " like '" + r + "' order by tblvals.time_pack;")
        """
        if ranks2 == []:
            sql_statement = (sql_statement + " = " + str(r) + " and tblvals.val > 0 order by tblvals.time_pack;")
        else:
            sql_statement = (sql_statement + " like '" + str(r) + "' and tblvals.val > 0 order by tblvals.time_pack;")
            #sql_statement = (sql_statement + " like '" + r + "' and tblvals.val > 0 order by tblvals.time_pack;")

        #params = [metric,r]
        #print "req_sql Executing query: ", sql_statement
        self.try_execute(c, sql_statement)
        #print "Done. "
        #print("Fetching rows.")
        #rows = c.fetchall()
        #if len(rows) <= 0:
        #    print("Error: query returned no rows.",)
        #    print(sql_statement, params)


  # Call demo with SQL statement given as argument and store standard output
  def req_sql2(self, c, metric):

    self.res_sql = ""
    sql_statement = ("SELECT value_name, value, time_pack FROM viewCombined WHERE value_name LIKE '" + metric+ "'")
    #sql_statement = ("SELECT * FROM viewCombined WHERE value_name LIKE '" + metric+ "'")
   
    print "sql statement: ", sql_statement 
    #self.try_execute(c, sql_statement)
    os.environ['SOS_SQL'] = sql_statement
    sos_bin_path = os.environ.get('SOS_BIN_DIR')
    print 'SOS BIN path: ', sos_bin_path
    os.system('cd '+ sos_bin_path)  
    print 'current dir: ', os.getcwd() 
    # Redirect stdout of passed command into a string
   
    soscmd = sos_bin_path + "/demo_app_silent --sql SOS_SQL"
    print 'soscmd: ', soscmd
    tmp_res_sql = subprocess.check_output(soscmd, shell=True)

    self.try_execute(c, sql_statement)

    #print 'stdout of SOS demo: ', sys.stdout
    #self.res_sql = resultstdout.getvalue()
    print 'tmp res_sql: ', tmp_res_sql   
    
    self.res_sql = tmp_res_sql.splitlines()
    # REmove first element of SQL result 
    self.res_sql.pop(0)

    for item_sql in self.res_sql:
      print 'res sql: ', item_sql 
      
 
  # Call demo with SQL statement given as argument and store standard output
  def req_sql(self, c, metric):

    self.res_sql = ""
    sql_statement = ("SELECT value_name, value, time_pack, max(frame) FROM viewCombined WHERE value_name LIKE '" + metric+ "' group by value_name;")
    #sql_statement = ("SELECT * FROM viewCombined WHERE value_name LIKE '" + metric+ "'")
   
    #print "sql statement: ", sql_statement 
    #self.try_execute(c, sql_statement)

    self.try_execute(c, sql_statement)

    
  def opendb(self):
    global min_timestamp
    # name of the sqlite database file
    #sqlite_file = arguments[0]

    # open the connection
    self.conn = self.open_connection()

    # get the number of ranks
    self.make_index(self.conn)
    self.ranks,self.procs = self.get_ranks(self.conn)
    while self.ranks.size == 0:
        time.sleep(1)
        self.ranks,self.procs = self.get_ranks(self.conn)
    print ("ranks: ", self.ranks)

    # get the number of nodes
    self.nodes,self.noderanks = self.get_nodes(self.conn)
    while self.nodes.size == 0:
        time.sleep(1)
        nodes,self.noderanks = self.get_nodes(self.conn)
    print ("nodes: ", self.nodes)

    self.get_min_timestamp_db(self.conn)
    #resize the figure
    # Get current size
    #fig_size = pl.rcParams["figure.figsize"]
    # Set figure width to 12 and height to 9
    #fig_size[0] = 12
    #fig_size[1] = 9
    #pl.rcParams["figure.figsize"] = fig_size
    #pl.ion()
    #docharts(c,nodes,noderanks,ranks,procs)
    #pl.tight_layout()

    print("Done.")

  def closedb(self):
    print("Closing connection to database.")
    # Closing the connection to the database file
    conn.close()
    #pl.tight_layout()

  def exec_sos_app(self):
    
    print 'SOS: Execute demo app'
    sos_path = os.environ.get('SOS_BUILD_DIR') 
    self.sos_bin_path = sos_path+"/bin"
    print 'SOS BIN PATH: ', self.sos_bin_path
    os.system("cd "+ self.sos_bin_path) 


  # Read and plot selected metrics coming from SOS 
  def readsosmetrics(self):

    print 'readsosmetrics'
    profile_t1 = time.time()

    self.opendb()

    print "metrics: ", self.metrics
    #self.get_min_timestamp()  
 
    while True:  
 
       #time.sleep(0.1)
       time.sleep(0.02)
       #print 'loop iteration ...'
       for i in range(self.ngraphs):
         #for i in range(self.nbsamples):
         if self.listRecordSample[i] != -1:
           j = self.listRecordSample[i]
     
           #print 'readsosmetrics: i=%d, j=%d' %(i,j)
           
           #rank = self.ranks[j]
           #rank2 = self.ranks2[j]
           #group_column = self.groupcolumns[j]
 	   metric = self.metrics[j]                   
           #print 'selected metric: ', metric 
           #print("Fetching rows.")
           #self.rows[j] = self.conn.fetchall()
           self.req_sql(self.conn,metric)
           #self.rows[j] = self.res_sql
           self.rows[j] = self.conn.fetchall()

           #print 'rows: ', self.rows[j]
           if len(self.rows[j]) <= 0:
             dummyvar  = 1
             #print("Error: query returned no rows.",)
           else:
             goodrecord = 1
         
           countsamples = 0
           for sample in self.rows[j]:
             params['ts'] = 0
             #print 'sample: ', sample
             #self.req_sql(self.conn, self.ranks, self.rows)
             profile_t2 = time.time()
             self.lock.acquire()
             #listsample = sample.split(',')
             #self.updateguisos(params,i,j,listsample)
             self.updateguisos_db(params,i,j,sample)
             self.lock.release()
             countsamples += 1

    self.closedb()

 
  def readsosmetrics_db(self):

     print 'readsosmetrics'
     profile_t1 = time.time()
     # Comment the method just for debugging purpose
     self.opendb() 
    
     print 'after opening db, read db and plot ....'
 
     while True:

       #print 'loop iteration ...'
       for i in range(self.ngraphs):
         #for i in range(self.nbsamples):
         if self.listRecordSample[i] != -1:
           j = self.listRecordSample[i]
     
           print 'readsosmetrics: i=%d, j=%d' %(i,j)
           
           #rank = self.ranks[j]
           #rank2 = self.ranks2[j]
           group_column = self.groupcolumns[j]
 	   metric = self.metrics[j]                   
           
           if metric == "Iteration": 
             self.req_sql_db(self.conn, self.ranks, [], group_column, metric)
           elif (metric == "CPU System%") or (metric == "CPU User%") or (metric == "Package-0 Energy"):
             self.req_sql_db(self.conn, self.nodes, self.noderanks, group_column, metric)
           elif (metric == "Matrix Size") or (metric == "status:VmHWM"):
             self.req_sql_db(self.conn, self.procs, [], group_column, metric)
          
           #print("Fetching rows.")
           self.rows[j] = self.conn.fetchall()
           #print 'rows: ', self.rows[j]
           if len(self.rows[j]) <= 0:
             print("Error: query returned no rows.",)
           else:
             goodrecord = 1
         
           countsamples = 0
           for sample in self.rows[j]:
             params['ts'] = 0
             #self.req_sql(self.conn, self.ranks, self.rows)
             profile_t2 = time.time()
             self.lock.acquire()
             self.updateguisos_db(params,i,j,sample)
             self.lock.release()
             countsamples += 1

     self.closedb()


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
           print 'payload =',payload
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
         print 'event element', e
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
         self.updateguibeacon(params,e)
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

     print 'subSpawn: load beacon subscriber library'
     envlibpath = os.environ['PYCOOLR_LIBPATH']
     libarbjsonbeep = cdll.LoadLibrary(envlibpath+'/libarbitraryjsonbeepmulsub.so')

     #libarbjsonbeep.subscribe(4, "MEMORY", "NODE_POWER_WATTS","MPI_T_PVAR")
     # Spawn a process to launch a subscriber
     # procSubscribe = multiprocessing.Process(target=self.subscribe, args=(libarbjsonbeep,))
     #jobs.append(procSubscribe)
     #procSubscribe.start()
     #self.readDB()

     if self.tool == "beacon":
       print 'Selected tool: beacon' 
       try:
         thread.start_new_thread(self.subscribe,(libarbjsonbeep,))
         thread.start_new_thread(self.readEvents,(libarbjsonbeep,))
         #thread.start_new_thread(self.readEvents,(libarbjsonbeep,))
       except Exception as errThread:
         print "Error: unable to start thread: ", errThread

     elif self.tool == "sos":
       try:
         #thread.start_new_thread(self.subscribe,(libarbjsonbeep,))
         thread.start_new_thread(self.readsosmetrics,())
         #thread.start_new_thread(self.readsosmetrics_db,())
      
       except Exception as errThread:
         print 'Error: unable to start thread: ', errThread


     self.refresh_plot()
     #self.readEvents(libarbjsonbeep)

     #while True:
     #  time.sleep(1)
     #  print 'draw canvas'
     #  try:
     #    self.canvas.draw()
     #  except Exception as errDraw:
     #    print 'Error drawing canvas: ', type(errDraw), errDraw

  def btncvarsupdatefn(self):
        print "Update CVARS"

        cvars_comm_mode = os.environ["CVARS_COMM_MODE"]

        strcvarsmetrics = ""
        strcvarsvalues = ""

        for i in range(self.numselectedcvars):

          if self.selectedcvarsmetrics[i] == self.listcfgcvarsarray[0]:
            self.listcvarsarrayindex[i] = self.listcvarsarrayindexentry[i].get()
            strcvarsmetrics += self.selectedcvarsmetrics[i]
            strcvarsmetrics += "["
            strcvarsmetrics += self.listcvarsarrayindex[i]
            strcvarsmetrics += "]"

            self.selectedcvarsvalues[i] =  self.listcvarsentry[i].get()
            strcvarsvalues += self.selectedcvarsvalues[i]
            print 'numselectedcvars=%d, index=%d' % (self.numselectedcvars, i)
            if i+1 < self.numselectedcvars:
              strcvarsmetrics += ","
              strcvarsvalues += ","

          else:
            self.selectedcvarsvalues[i] =  self.listcvarsentry[i].get()
            strcvarsmetrics += self.selectedcvarsmetrics[i]
            strcvarsvalues += self.selectedcvarsvalues[i]
            #self.strcvars += self.selectedcvarsmetrics[i]
            #self.strcvars += "=" 
            #self.strcvars += self.selectedcvarsvalues[i]
            #strcvars += ","
            print 'numselectedcvars=%d, index=%d' % (self.numselectedcvars, i)
            if i+1 < self.numselectedcvars:
              strcvarsmetrics += ","
              strcvarsvalues += ","
              #self.strcvars += "," 

        self.strcvars = strcvarsmetrics
        self.strcvars += ";"
        #self.strcvars += ":"
        self.strcvars += strcvarsvalues

        print "strcvarsmetrics: ", strcvarsmetrics
        print "strcvarsvalues: ", strcvarsvalues
        print "strcvars: ", self.strcvars

        # Test if we have to communicate MPI_T CVARS in a Publish/Subscribe mode       
        if cvars_comm_mode == "pub":

          print "Publishing CVARS"

          pycoolr_inst_path=os.environ['PYCOOLR_INST_PATH']
          os.system("python "+pycoolr_inst_path+"/gui/pycoolr-plot/coolr-pub-cvars.py "+str(strcvarsmetrics)+" "+str(strcvarsvalues))

        # We communicate MPI_T CVARS using the Filesystem
        elif cvars_comm_mode == "fs":

          cvars_file = ""
          cvars_file = os.environ['MPIT_CVARS_PATH']
        #print "TAU CVARS file: ", cvars_file

          fcvars = open(cvars_file, "w")

          fcvars.write(self.strcvars+"\n")
        #fcvars.write(strcvarsvalues)


  def onselectcvars(self,evt):
        w = evt.widget
        selection = w.curselection()
        self.selectedcvar = w.get(selection[0])
        #print "selection:", selection, ": '%s'" % self.selectedcvar
        self.cvarsindexrow = 0

        self.numselectedcvars = len(selection)
        self.selectedcvarsmetrics = [None] * self.numselectedcvars
        self.selectedcvarsvalues = [None] * self.numselectedcvars

        for i in range(len(selection)):
          value = w.get(selection[i])
          print "selection:", selection, ": '%s'" % value
          self.selectedcvarsmetrics[i] = value

        if self.listlabelcvarsmetric:
          for i in range(len(self.listlabelcvarsmetric)):
            self.listlabelcvarsmetric[i].grid_forget()

        if self.listcvarsentry:
          for i in range(len(self.listcvarsentry)):
            self.listcvarsentry[i].grid_forget()

        if self.btncvarsupdate:
          self.btncvarsupdate.grid_forget()

        listintselection = [int (i) for i in selection]

        self.listlabelcvarsmetric = [None] * len(selection)
        self.listcvarsentry = [None] * len(selection)
        self.listlabelcvarsarrayindex = [None] * len(selection)
        self.listcvarsarrayindexentry = [None] * len(selection)
        self.listcvarsarrayindex = [None] * len(selection)

        print 'selection: ', selection
        print 'range selection: ', range(len(selection))

        for cvaritem, cvarindex in zip(selection, range(len(selection))):

          value = w.get(selection[cvarindex])
          print 'len selection: ', len(selection)
          print 'value of item %d: %s ' % (cvarindex, value)
          print 'cvaritem: ', cvaritem
          print 'cvarindex= ', cvarindex
          print 'cvarsindexrow= ', self.cvarsindexrow

          print 'cfg cvars array:', self.listcfgcvarsarray[0]
          if value == self.listcfgcvarsarray[0]:

            self.listlabelcvarsmetric[cvarindex]=Tk.Label(self.stepCvarsUpdate,  text=value)
            self.listlabelcvarsmetric[cvarindex].grid(row=self.cvarsindexrow,column=3,sticky='NS')

            self.listcvarsentry[cvarindex] = Tk.Entry(self.stepCvarsUpdate)
            self.listcvarsentry[cvarindex].grid(row=self.cvarsindexrow+1, column=3, sticky = Tk.E+ Tk.W)

            self.listlabelcvarsarrayindex[cvarindex]=Tk.Label(self.stepCvarsUpdate,  text="CVAR Array index")
            self.listlabelcvarsarrayindex[cvarindex].grid(row=self.cvarsindexrow+2,column=3,sticky='NS')

            self.listcvarsarrayindexentry[cvarindex] = Tk.Entry(self.stepCvarsUpdate)
            self.listcvarsarrayindexentry[cvarindex].grid(row=self.cvarsindexrow+3, column=3, sticky = Tk.E+ Tk.W)

            self.cvarsindexrow += 4

          else:

            self.listlabelcvarsmetric[cvarindex]=Tk.Label(self.stepCvarsUpdate, text=value)
            self.listlabelcvarsmetric[cvarindex].grid(row=self.cvarsindexrow,column=3,sticky='NS')

            self.listcvarsentry[cvarindex] = Tk.Entry(self.stepCvarsUpdate)
            self.listcvarsentry[cvarindex].grid(row=self.cvarsindexrow+1, column=3, sticky = Tk.E+ Tk.W)

            self.cvarsindexrow += 2

        self.btncvarsupdate = Tk.Button(self.stepCvarsUpdate,text="Update",command=self.btncvarsupdatefn)
        self.btncvarsupdate.grid(row=self.cvarsindexrow, column=3, sticky="we")

        #self.labelcvarscontent=Tk.Label(self.f2, text=self.selectedcvar).grid(row=0,column=2, sticky="nw")
        #self.labelcvarscontent=Tk.Label(self.stepCvarsUpdate, text=self.selectedcvar)
        #self.labelcvarscontent.config(text=self.selectedcvar)

        self.cvarsmetricvar.set(self.selectedcvar)
        #self.cvarsentry.delete(0,Tk.END)
        #self.cvarsentry.insert(0,str(self.dictcvars[self.selectedcvar]))

        #self.cvarswin.after(1000, self.onselectcvars)

  def onselectmetrics(self,evt):
        w = evt.widget
        selection = w.curselection()
        for i in range(len(selection)):
          value = w.get(selection[i])
          #print "selection:", selection, ": '%s'" % value
         
        listintselection = [int (i) for i in selection]
        print 'listintselection: ', listintselection

        for i in range(self.nbsamples):
          if (self.listSamplesAllocated[i] > -1) and (i not in listintselection):
            self.listSamplesAllocated[i] = -1

        for i in range(self.ngraphs):
         
          if (self.listUsedGraphs[i] not in listintselection) or (self.listUsedGraphs[i] == -1):
            
            for j in listintselection:
        
              if self.listSamplesAllocated[j] == -1:    
                #index = int(j)
                self.listUsedGraphs[i] = j
                print 'graph %d allocated to sample %d' % (i, j)
                self.listRecordSample[i] = j
                self.listSamplesAllocated[j] = i
                break

  def onselectpvars2(self,evt):
	# Note here that Tkinter passes an event object to onselect()
        #print 'selected item of list: ', self.l1.get(self.l1.curselection()) 
    	w = evt.widget
        selection = w.curselection()
        for i in range(len(selection)):
          value = w.get(selection[i])
          #print "selection:", selection, ": '%s'" % value

        for i in selection:
          #value = w.get(selection[i])
          index = int(i)
          if self.listSamplesAllocated[index] == -1:

            for j in range(self.ngraphs):
      
              if self.listUsedGraphs[j] == 0:

                # Mark current graph as used
                self.listUsedGraphs[j] = 1           
                # Record the current graph as plotting the current sample
                print 'Record Sample %d for graph %d' %(index,j)
                self.listRecordSample[j] = index

                # Mark current sample as allocated to the current graph
                #if self.listSamplesAllocated[index] == -1:
                self.listSamplesAllocated[index] = j

                # Associate current graph to the current sample
                self.listSamplesGraphs[index] = j
                # Consider next sample 
                break

    	#value = w.get(selection[0])
        #print "selection:", selection, ": '%s'" % value
    	#index = int(w.curselection()[0])
    	#value = w.get(index)
    	#print 'You selected item %d: "%s"' % (index, value)


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


#  def eventmenu():  
#     print 'nothing'
      

root = Tkinter.Tk()

app = Coolrsub(root,2,3)
root.mainloop()

  
