#!/usr/bin/env python

from listrotate import *

from clr_matplot_graphs import *

# the class name should match with the module name
class graph_power:
    def __init__(self, params, layout):
        self.npkgs = params['info']['npkgs']
        self.data_lr = {}
        self.data_lr['pkg'] = [listrotate2D(length=params['lrlen']) for i in range(self.npkgs)]
        self.data_lr['dram'] = [listrotate2D(length=params['lrlen']) for i in range(self.npkgs)]

        self.ax = layout.getax()

        
    def update(self, params, sample):
        if sample['node'] == params['targetnode'] and sample['sample'] == 'energy':

            t = sample['time'] - params['ts']
            params['cur'] = t # this is used in update()

            for pkgid in range(self.npkgs):
                tmppow = sample['power']['p%d'%pkgid]
                if tmppow < 0:
                    print
                    print 'WARNING: power is negative. Check %s' % (tmppow, params['cfg']['outputfn'])
                    print
                    return
                tmplim = sample['powercap']['p%d'%pkgid]
                tmppowdram =  sample['power']['p%d/dram'%pkgid]
                self.data_lr['pkg'][pkgid].add(t, tmppow, tmplim)
                self.data_lr['dram'][pkgid].add(t, tmppowdram)

            #
            # drawing
            #
            gxsec = params['gxsec']
            cfg = params['cfg']

            self.ax.cla()
            self.ax.axis([t-gxsec, t, cfg['pwrmin'], cfg['pwrmax']])

            pkgid = 0
            for t in self.data_lr['pkg']:
                x = t.getlistx()
                y = t.getlisty()
                ycap = t.getlisto()
                self.ax.plot(x,ycap, color='red', label='PKG%dlimit'%pkgid )
                self.ax.plot(x,y,color=params['pkgcolors'][pkgid], label='PKG%d'%pkgid)
                pkgid += 1

            pkgid = 0
            for t in self.data_lr['dram']:
                x = t.getlistx()
                y = t.getlisty()
                self.ax.plot(x,y,color=params['pkgcolors'][pkgid], linestyle='--', label='PKG%ddram'%pkgid)
                pkgid += 1

            self.ax.legend(loc='lower left', prop={'size':9})
            self.ax.set_xlabel('Time [s]')
            self.ax.set_ylabel('Power [W]')
            self.ax.set_title("Node Power (%s)" % params['targetnode'])
