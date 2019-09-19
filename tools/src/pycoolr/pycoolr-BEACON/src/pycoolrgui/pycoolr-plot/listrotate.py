#!/usr/bin/env python

#
# listrotate1D and listrotate2D are convenient data structure, which
# is a fixed-size list that returns a rotated list from the oldest
# item. It is convenient when we plot a graph using matplotlib in an
# animated way while querying data. See below "__main__" for example.
#
# Kaz Yoshii <kazutomo.yoshii@gmail.com>
#

import numpy as np

class listrotate1D:
    def __init__(self, length = 400):
        self.length = length
        self.x = [ np.nan for i in range(length) ]
        self.r = [ np.nan for i in range(length) ]
        self.pos = 0

    def add(self, xv):
        self.x[self.pos] = xv
        self.pos += 1
        if self.pos == self.length:
            self.pos = 0
        xm1 = self.x[self.pos-1]
        xm2 = self.x[self.pos-2]
        if not np.nan in (xm1, xm2):
            self.r[self.pos-1] = (xm1-xm2)

    def getlast(self):
        return self.x[self.pos-1]

    def getlastr(self):
        return self.r[self.pos-1]
            
    def getlist(self):
        if self.pos == 0:
            return self.x
        return self.x[self.pos:] + self.x[0:self.pos]

    def getlistr(self):
        if self.pos == 0:
            return self.r
        return self.r[self.pos:] + self.r[0:self.pos]


class listrotate2D:
    def __init__(self, length = 400):
        self.length = length
        self.x = [ np.nan for i in range(length) ]
        self.y = [ np.nan for i in range(length) ]
        self.r = [ np.nan for i in range(length) ] # rate
        self.o = [ np.nan for i in range(length) ] # option values
        self.pos = 0

    def add(self, xv, yv, ov=None): # ov: option values
        self.x[self.pos] = xv
        self.y[self.pos] = yv
        self.o[self.pos] = ov
        self.pos += 1
        if self.pos == self.length:
            self.pos = 0

        xm1 = self.x[self.pos-1]
        xm2 = self.x[self.pos-2]
        ym1 = self.y[self.pos-1]
        ym2 = self.y[self.pos-2]
        if not np.nan in (xm1, xm2, ym1, ym2):
            if (xm1-xm2)==0.0:
                self.r[self.pos-1] = 0.0
            else:
                self.r[self.pos-1] = (ym1-ym2)/(xm1-xm2)

    def getlastx(self):
        return self.x[self.pos-1]
    def getlasty(self):
        return self.y[self.pos-1]
    def getlastr(self):
        return self.r[self.pos-1]
    def getlasto(self):
        return self.o[self.pos-1]
            
    def getlistx(self):
        if self.pos == 0:
            return self.x
        return self.x[self.pos:] + self.x[0:self.pos]
    def getlisty(self):
        if self.pos == 0:
            return self.y
        return self.y[self.pos:] + self.y[0:self.pos]
    def getlistr(self):
        if self.pos == 0:
            return self.r
        return self.r[self.pos:] + self.r[0:self.pos]
    def getlisto(self):
        if self.pos == 0:
            return self.o
        return self.o[self.pos:] + self.o[0:self.pos]

    def getmaxy(self):
        if np.all(np.isnan(self.y)):
            return 0.0
        else :
            return np.nanmax(self.y)
   
    def getminy(self):
        if np.all(np.isnan(self.y)):
            return 0.0
        else :
            return np.nanmin(self.y)
 
if __name__ == '__main__':

    lr = listrotate2D(5)
    for i in range(8):
        lr.add(i,i)
        print lr.getlistx()
        print lr.getlisty()
        print lr.getlistr()
        print 

    print '------------'
    lr = listrotate1D(5)
    for i in range(8):
        lr.add(i)
        print lr.getlist()
        print lr.getlistr()
        print 
