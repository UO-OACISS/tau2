#!/usr/bin/env python

from listrotate import *

from clr_matplot_graphs import *

# the class name should match with the module name
class graph_freq:
    def __init__(self, params, layout):

        npkgs = params['info']['npkgs']
        self.freq_lr = [listrotate2D(length=params['lrlen']) for i in range(npkgs)]
        self.ax = layout.getax()

    def update(self, params, sample):
        if sample['node'] == params['targetnode'] and sample['sample'] == 'freq':
            #
            # data handling
            #
            t = sample['time'] - params['ts']

            for p in range(params['info']['npkgs']):
                v0 = sample['p%d' % p]['mean']
                v1 = sample['p%d' % p]['std']
                self.freq_lr[p].add(t,v0,v1)

            #
            # graph handling : line+errbar
            #
            pdata = self.freq_lr
            gxsec = params['gxsec']

            self.ax.cla()
            self.ax.set_xlim([t-gxsec, t])

            cfg = params['cfg']
            self.ax.set_ylim([cfg['freqmin'], cfg['freqmax']])
            self.ax.axhspan(cfg["freqnorm"], cfg["freqmax"], facecolor='#eeeeee' ) #, alpha=0.1)

            pdata = self.freq_lr
            pkgid = 0
            for t in pdata:
                x = t.getlistx()
                y = t.getlisty()
                e = t.getlisto()
                self.ax.plot(x,y,scaley=True,color=params['pkgcolors'][pkgid], label='PKG%d'%pkgid)
                self.ax.errorbar(x,y,yerr=e, lw=.2, color=params['pkgcolors'][pkgid], label = '')
                pkgid += 1

            self.ax.set_xlabel('Time [s]')
            self.ax.set_ylabel('Active Frequency [GHz]')
            self.ax.set_title('Node Frequency (%s)' % params['targetnode'] )

            self.ax.legend(loc='lower left', prop={'size':9})
