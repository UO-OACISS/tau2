#!/usr/bin/env python

import os
import thread
import time
import ctypes
from ctypes import cdll
from ctypes.util import find_library

#libmonitor = ctypes.cdll.LoadLibrary(find_library('test'))
#libmonitor = cdll.LoadLibrary('/home/users/aurelem/beacon/gui/pycoolr-orig/libbeaconmonitor.so')
#libmonitor = cdll.LoadLibrary(os.getcwd()+'/libbeaconmonitor.so')
libarbjsonbeep = cdll.LoadLibrary(os.getcwd()+'/libarbitraryjsonbeep.so')
#libmonitor = ctypes.cdll.LoadLibrary(find_library('beaconmonitor'))

nb_events = 0
tmp_nb_events = 0
listEvents = []
events_index = 0
#low_index = 0
#high_index = 0

def Subscribe():
  print 'start thread with Subscribe'
  print 'lib arbitrary json beep shared lib loaded:', libarbjsonbeep

  #libmonitor.init_table()
  #libmonitor.monitor(2, "MPI_T_PVAR")
  libarbjsonbeep.subscribe(2, "MPI_T_PVAR")

def readEvents():
 #read_events = libmonitor.read_data_events
 #read_events.restype = ctypes.POINTER(store_event_t)
 low_index = 0
 high_index = 0
 tmp_nb_events = 0
 update = 0

 while True:

  #read_ts = libmonitor.read_ts
  #read_ts.argtypes = ()
  #read_ts.restype = ctypes.c_double

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
  
  print 'Number of events', nb_events
 
  #events_index = nb_events-1
  if high_index != nb_events-1 and high_index != 0:
   low_index = high_index+1

  high_index = nb_events-1

  print 'low_index='+str(low_index)+' high_index='+str(high_index)

  if update == 1:
   read_payload = libarbjsonbeep.read_payload
   read_payload.argtype = (ctypes.c_int)
   read_payload.restype = ctypes.POINTER(ctypes.c_char)

   for i in range(low_index, high_index+1):
    #resultPayload = read_payload(1)
    #resultPayload = read_payload(events_index)
    resultPayload = read_payload(i)
    print 'Print event '+str(i)
    #print 'stored events:', store_events
    #print 'stored ts:', result

    payload = ""
    for j in range(0,2048):
     payload += resultPayload[j]

    listEvents.append(payload)

    #for i in range(0,10):
    print 'stored payload:', payload
 
   tmp_nb_events = nb_events
  #low_index = nb_events

  time.sleep(2)

try:
 thread.start_new_thread(Subscribe,()) 
 thread.start_new_thread(readEvents,()) 
except:
 print "Error: unable to start thread"


while 1: 
 pass
#events_data = libmonitor.sub_events 

#libmonitor.test_py()

