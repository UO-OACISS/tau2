#!/usr/bin/env python

import os
import ctypes
from ctypes import cdll
from ctypes.util import find_library

#libmonitor = ctypes.cdll.LoadLibrary(find_library('test'))
#libmonitor = cdll.LoadLibrary('/home/users/aurelem/beacon/gui/pycoolr-orig/libbeaconmonitor.so')
#libmonitor = cdll.LoadLibrary(os.getcwd()+'/libbeaconmonitor.so')
libmonitor = cdll.LoadLibrary(os.getcwd()+'/libbeaconmonitormulsub.so')
#libmonitor = ctypes.cdll.LoadLibrary(find_library('beaconmonitor'))

print 'lib beacon monitor mul sub shared lib loaded:', libmonitor

libmonitor.init_table()
libmonitor.monitor(3, "MEMORY MPI_T_PVAR")

events_data = libmonitor.sub_events 

#libmonitor.test_py()

