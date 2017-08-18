#!/usr/bin/env python

import os
import sys
import thread
import time
import ctypes
from ctypes import cdll
from ctypes.util import find_library

from init_coolr import *
from graph_beacon import *
from layout import *
#libarbjsonbeep = cdll.LoadLibrary(os.getcwd()+'/libarbitraryjsonbeep.so')

#class beaconSubscribe:

def subscribe(libarbjsonbeep):
     print 'start thread with Subscribe'
     print 'lib arbitrary json beep shared lib loaded:', libarbjsonbeep

     #libmonitor.init_table()
     #libmonitor.monitor(2, "MPI_T_PVAR")
     libarbjsonbeep.subscribe(2, "MPI_T_PVAR") 

def readEvents(libarbjsonbeep):

     print 'start thread readEvents'
     layout = layoutclass(cfg["nrows"], cfg["ncols"])
     modulelist = [] # a list of graph modules

     for k in cfg["modnames"]:
       name='graph_%s' % k
       m = __import__(name)
       c = getattr(m, name)
       modulelist.append( c(params, layout) )

     #read_events = libmonitor.read_data_events
     #read_events.restype = ctypes.POINTER(store_event_t)
     low_index = 0
     high_index = 0
     tmp_nb_events = 0
     update = 0
     listEvents = []

     while True:

       profile_t1 = time.time()
       #read_ts = libmonitor.read_ts
       #read_ts.argtypes = ()
       #read_ts.restype = ctypes.c_double

       params['ts'] = 0
       #result = read_ts()

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

       if update == 1:
         read_payload = libarbjsonbeep.read_payload
         read_payload.argtype = (ctypes.c_int)
         read_payload.restype = ctypes.POINTER(ctypes.c_char)
    
         #print 'low_index:'+str(low_index)+' high_index:'+str(high_index)
         for i in range(low_index, high_index+1):
           #resultPayload = read_payload(1)
           #resultPayload = read_payload(events_index)
           resultPayload = read_payload(i)

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
           listEvents.append(j)

           #for i in range(0,10):
           #print 'stored payload:', payload
 
       tmp_nb_events = nb_events
       #low_index = nb_events

       #time.sleep(2)

       #print 'Test listEvents: ', listEvents
       if not listEvents:
         continue
      #print 'no listEvents'

       #print 'parsing listEvents for plotting'
       for e in listEvents:
         #iterator += 1
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

        #if iterator%2 == 0:

         #print 'iterate coolr display loop: before updating'
         #for m in modulelist:
          #       m.update(params,e)
                #print 'module: '+str(m)
         #updateplot(params,e)

       #draw to refresh plotting
       layout.canvas.draw()
       #plt.draw()

       profile_t3 = time.time()
       #print 'iterate coolr display loop: t3='+str(profile_t3)

       pausesec = 0.0
       if intervalsec > profile_t3-profile_t1:
         pausesec = intervalsec - (profile_t3-profile_t1)
       if pausesec > 0.0:
         plt.pause(pausesec)

       #print 'Profile Time [S]: %.2lf (%.2lf+%.2lf+%.2lf) / Queried %3d items from DB' %\
       #    (profile_t3-profile_t1+pausesec, profile_t2-profile_t1,\
       #     profile_t3-profile_t2, pausesec, len(j))


def subSpawn():

     print 'load beacon subscriber library'
     libarbjsonbeep = cdll.LoadLibrary(os.getcwd()+'/libarbitraryjsonbeep.so')
     #libmonitor = ctypes.cdll.LoadLibrary(find_library('beaconmonitor'))

     nb_events = 0
     tmp_nb_events = 0
     #listEvents = []
     events_index = 0
     #low_index = 0
     #high_index = 0

     try:
       thread.start_new_thread(subscribe,(libarbjsonbeep,)) 
       thread.start_new_thread(readEvents,(libarbjsonbeep,)) 
     except Exception as errThread:
       print "Error: unable to start thread: ", errThread

     while 1: 
       pass



 
