#!/usr/bin/env python

import pytau
from time import sleep

x = pytau.profileTimer("A Sleep for excl 5 secs")
y = pytau.profileTimer("B Sleep for excl 2 secs")
pytau.start(x)
print "Sleeping for 5 secs ..."
sleep(1)
pytau.start(y)
print "Sleeping for 2 secs ..."
z = 3 
sleep(1)
pytau.stop(y)
pytau.dbDump()
pytau.stop(x)

