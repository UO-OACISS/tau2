#!/usr/bin/env python

from listrotate import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# the class name should match with the module name
class graph_tau:
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
        self.titles = ('Memory Footprint', 'Peak Memory', 'Power', 'MV2 Memory allocated', 'MV2 Num malloc calls','MV2 num free calls')
        #self.colors = ('blue', 'red', 'green')
        #self.colors = ('blue', 'red', 'green','blue','red','green','blue','red','green')
        self.colors = ('blue', 'red', 'green','blue','red','green')

        #self.titles = ('Power')
        #self.colors = ('green')
        
        self.data_lr = [listrotate2D(length=params['lrlen']) for i in range(self.ngraphs)]
        self.ax = [layout.getax() for i in range(self.ngraphs)]
        self.ytop = [1 for i in range(self.ngraphs)]
        self.ybot = [1 for i in range(self.ngraphs)] 

    def update(self, params, sample):
        if self.ngraphs == 0:
            return

	print 'starting update'
        if sample['node'] == params['targetnode'] and sample['sample'] == 'tau':
            #
            # data handling
            #
            t = sample['time'] - params['ts']
	    goodrecord=0
            for i in range(self.ngraphs):
		ref1=params['cfg']
		ref2=ref1['appsamples']
		ref3=ref2[i]
		if ref3 in sample:
			ref4=sample[ref3]
                	self.data_lr[i].add(t,ref4)
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

            for i in range(self.ngraphs):
                ax = self.ax[i]
                pdata = self.data_lr[i]
                label = params['cfg']['appsamples'][i]
                ax.cla()
                ax.set_xlim([t-gxsec, t])

                x = pdata.getlistx()
                y = pdata.getlisty()

                ymax = pdata.getmaxy()
                ymin = pdata.getminy()
                if ymax > self.ytop[i]:
                    self.ytop[i] = ymax * 1.1

		if self.ybot[i] == 1 or ymin < self.ybot[i]:
		    self.ybot[i] = ymin*.9

                ax.set_ylim([self.ybot[i], self.ytop[i]])
                ax.ticklabel_format(axis='y', style='sci', scilimits=(1,0))

                #ax.plot(x, y, label='', color=self.colors[i], lw=1.2)
                ax.plot(x, y, label='', 'ro')
                #ax.bar(x, y, width = .6, edgecolor='none', color='#77bb88' )
                #ax.plot(x,y, 'ro', scaley=True, label='')

                ax.set_xlabel('Time [s]')
                ax.set_ylabel(label)

                ax.set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )

        else:
            t = sample['time'] - params['ts']
            gxsec = params['gxsec']

            for i in range(self.ngraphs):
                self.ax[i].set_xlim([t-gxsec, t])
                self.ax[i].set_title('%s: %s (%s)' % (params['cfg']['appname'], self.titles[i], params['targetnode']) )
	print 'ending update'

