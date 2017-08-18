#!/usr/bin/env python

from listrotate import *

from clr_matplot_graphs import *

# the class name should match with the module name
class graph_enclave:
    def __init__(self, params, layout):
        self.n_enclaves = len( params['enclaves'] )
        self.data_lr = [listrotate2D(length=params['lrlen']) for i in range(self.n_enclaves)]

        # multiple lines in one graph
        self.ax = layout.getax()
        self.ytop = 1

    def update(self, params, sample):
        if not sample['node'] in params['enclaves']:
                   return

        # assume that enclave samples always have the 'energy' tag

        t = sample['time'] - params['ts']
        params['cur'] = t # this is used in update()

        # update data
        for i in range(self.n_enclaves):
            if sample['node'] == params['enclaves'][i]:
                tmp = sample['power']['total']
                self.data_lr[i].add(t, tmp )

        #
        # drawing
        #
        gxsec = params['gxsec']
        cfg = params['cfg']

        colors = ['#aaaa22', '#22aaaa', '#aa22aa', '#aaaa222' ]

        ax = self.ax

        ax.cla()
        ax.set_xlim([t-gxsec, t])

        for i in range(self.n_enclaves):
            ymax = self.data_lr[i].getmaxy()
            if ymax > self.ytop:
                self.ytop = ymax * 1.1

        ax.set_ylim([0, self.ytop])

        for i in range(self.n_enclaves):
            x = self.data_lr[i].getlistx()
            y = self.data_lr[i].getlisty()
            self.ax.plot(x, y, label=params['enclaves'][i], color=colors[i])

        self.ax.legend(loc='lower left', prop={'size':9})
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('Power [W]')
        self.ax.set_title("Enclave Power")
