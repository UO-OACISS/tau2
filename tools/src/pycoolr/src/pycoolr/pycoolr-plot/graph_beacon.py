#!/usr/bin/env python

from listrotate import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# the class name should match with the module name
class graph_beacon:
    def __init__(self, params, layout):
        #self.ngraphs = 0
        self.ngraphs = 6
        #self.ngraphs = 3

        if not params['cfg'].has_key('appsamples'):
            return

        self.ngraphs = len(params['cfg']['appsamples'])
        print 'samples', params['cfg']['appsamples']

#        self.titles = ('Memory Footprint', 'Peak Memory')
#        self.colors = ('blue', 'red')

        #self.titles = ('Memory Footprint', 'Peak Memory', 'Power')
        #self.titles = ('Memory Footprint', 'Peak Memory', 'Power', 'MV2 Memory allocated', 'MV2 Num malloc calls','MV2 num free calls','MV2 eager avail buffer', 'MV2 rndv avail buffer','MV2 eager sent buffer')
        self.titles = ('Memory Footprint (VmRSS - Resident Set Size)', 'Peak Memory Usage (VmHWM - High Water Mark)', 'rapl::RAPL_ENERGY_PKG (CPU Socket Power in Watts)', 'mem_allocated (Current level of allocated memory within the MPI library)', 'num_malloc_calls (Number of MPIT_malloc calls)','num_free_calls (Number of MPIT_free calls)')
        #self.titles = ('Memory Footprint', 'Peak Memory', 'Power', 'MV2 Memory allocated', 'MV2 Num malloc calls','MV2 Num free calls')
        #self.titles = ('Memory Footprint', 'Peak Memory', 'Power', 'MV2 Memory allocated', 'MV2 Num malloc calls')
        #self.colors = ('blue', 'red', 'green')
        #self.colors = ('blue', 'red', 'green','blue','red','green','blue','red','green')
        self.colors = ('blue', 'red', 'green', 'blue', 'red', 'green')
        #self.colors = ('blue', 'red', 'green', 'blue', 'red')

        #self.titles = ('Power')
        #self.colors = ('green')
        
        self.data_lr = [listrotate2D(length=params['lrlen']) for i in range(self.ngraphs)]
        self.ax = [layout.getax() for i in range(self.ngraphs)]
        self.ytop = [1 for i in range(self.ngraphs)]
        self.ybot = [1 for i in range(self.ngraphs)] 

    def update(self, params, sample):
        if self.ngraphs == 0:
            return

        mean_val = 0
        total_val=0
        num_vals=0

	#print 'starting update'
        if sample['node'] == params['targetnode'] and sample['sample'] == 'tau':
            #
            # data handling
            #
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
                        #print 'display record ref4='+str(ref4)
                	self.data_lr[i].add(t,ref4)
                	#self.data_lr[i].add(t,mean_val)
			goodrecord=1

	    if goodrecord==0:
#		print 'bad record'
		return

            #
            # graph handling
            #
            gxsec = params['gxsec']
            #
            #

            #print 'parse graphs'
            for i in range(self.ngraphs):
                ax = self.ax[i]
                pdata = self.data_lr[i]
                #label = params['cfg']['appsamples'][i]
                label = params['cfg']['units'][i]
                ax.cla()
                ax.set_xlim([t-gxsec, t])

                #print 'get x and y'
                x = pdata.getlistx()
                y = pdata.getlisty()

                #print 'get ymax and ymin'
                ymax = pdata.getmaxy()
                ymin = pdata.getminy()
                if ymax > self.ytop[i]:
                    self.ytop[i] = ymax * 1.1

		if self.ybot[i] == 1 or ymin < self.ybot[i]:
		    self.ybot[i] = ymin*.9

                ax.set_ylim([self.ybot[i], self.ytop[i]])
                ax.ticklabel_format(axis='y', style='sci', scilimits=(1,0))

                #print 'ax plot'
                #ax.plot(x, y, label='', color=self.colors[i], lw=1.2)
                ax.plot(x, y, 'rs', lw=2)
                #ax.bar(x, y, width = .6, edgecolor='none', color='#77bb88' )
                #ax.plot(x,y, 'ro', scaley=True, label='')

                #print 'ax set x y label'
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(label)

                #print 'ax set title'
                ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )

        else:
            t = sample['time'] - params['ts']
            gxsec = params['gxsec']

            for i in range(self.ngraphs):
                self.ax[i].set_xlim([t-gxsec, t])
                self.ax[i].set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )
	#print 'ending update'

