#
# matplot graph class definition
#
# backend independent implementation
#
# Kazutomo Yoshii <ky@anl.gov>
#

import os, sys

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
#import matplotlib.collections as collections
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png

from listrotate import *

class plot_info:
    def ypos(self,i):
        return 1.0 - 0.05*i

    def __init__(self, ax, params):
        self.ax = ax

        dir=os.path.abspath(os.path.dirname(sys.argv[0]))

        ax.axis([0,1,0,1])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on=True

        cfg = params['cfg']
        info = params['info']

        xoff = 0.05

        l=1
#        for p in range(info['npkgs']):
#           ax.text( 0.1, self.ypos(l), 'CPU PKG%d' % p, color=params['pkgcolors'][p] )
#            l += 1
#        l += 2

        ax.text(xoff, self.ypos(l), '[INFO]' )
        l += 2
        ax.text(xoff, self.ypos(l), 'Description : %s' % cfg['desc'] )
        l += 1
        ax.text(xoff, self.ypos(l), 'Node : %s' % info['nodeinfo'] )
        l += 1
        ax.text(xoff, self.ypos(l), 'Linux kernel : %s' % info['kernelversion'] )
        l += 1
        plt.text( xoff, self.ypos(l), 'Freq. driver : %s' % info['freqdriver'] )
        l += 1
        plt.text( xoff, self.ypos(l), 'Memory [GB] : %d' % (int(info['memoryKB'])/1024/1024) )
        l += 1
        plt.text( xoff, self.ypos(l), 'CPU model : %s' % info['cpumodel'] )
        l += 1
        plt.text( xoff, self.ypos(l), 'No of procs : %s' % info['ncpus'] )
        l += 1
        plt.text( xoff, self.ypos(l), 'No of pkgs : %s' % info['npkgs'] )
        l += 1
        plt.text( xoff, self.ypos(l), 'No of NUMA nodes: %d' % info['nnodes'] )

        a = info['pkg0phyid']
        ht = 'enabled'
        if len(a) == len(set(a)):
            ht = 'disabled'
        l += 1
        plt.text( xoff, self.ypos(l), 'Hyperthread : %s' % ht)
#        l += 1
#        plt.text( xoff, ypos(l), 'Powercap pkg0 : %d Watt' % s_powercap['p0'] )
#        l += 1
#        plt.text( xoff, ypos(l), 'Powercap pkg1 : %d Watt' % s_powercap['p1'] )
#        l += 1


        fn = get_sample_data("%s/coolr-logo-poweredby-48.png" % dir, asfileobj=False)
        arr = read_png(fn)
        imagebox = OffsetImage(arr, zoom=0.6)
        ab = AnnotationBbox(imagebox, (0, 0),
                            xybox=(.8, .1),
                            xycoords='data',
                            boxcoords="axes fraction",
                            pad=0.5)
        ax.add_artist(ab)

    def update(self, n):
        # self.text1.set_text('%d' % n)
        s = 'dummy'


class plot_totpwr:
    def __init__(self, ax, params, totpwrs):
        self.ax = ax
        self.update(params, totpwrs)

    def update(self, params, totpwrs):

        cfg = params['cfg']
        cur_t = params['cur']
        gxsec = params['gxsec']

        self.ax.cla() # this is a brute-force way to update

        self.ax.axis([cur_t-gxsec, cur_t, cfg['pwrmin'], cfg['acpwrmax']]) # [xmin,xmax,ymin,ymax]

        x = totpwrs[0].getlistx()
        y = totpwrs[0].getlisty()
        self.ax.plot(x,y, scaley=False, color='black', label='RAPL total' )
        if len(totpwrs) > 1:
            x = totpwrs[1].getlistx()
            y = totpwrs[1].getlisty()
            self.ax.plot(x,y, '--', scaley=False,  color='black', label='AC' )

        self.ax.legend(loc='lower left', prop={'size':9})
        self.ax.set_xlabel('Time [S]')
        self.ax.set_ylabel('Power [W]')


# the following plots can be generalized
class plot_xsbench:
    def __init__(self, ax, params, lps):
        self.ax = ax

        # too lazy to figure out axhspan's object. fix this later
        self.update(params, lps)

    def update(self, params, lps):

        cfg = params['cfg']
        cur_t = params['cur']
        gxsec = params['gxsec']

        self.ax.cla() # this is a brute-force way to update

        self.ax.set_xlim([cur_t-gxsec, cur_t])
        self.ax.set_ylim(bottom=0)

        x = lps.getlistx()
        y = lps.getlisty()
        #self.ax.plot(x,y, scaley=False, color='black', label='' )
        self.ax.bar(x,y,  color='black', label='' )

        self.ax.legend(loc='lower left', prop={'size':9})
        self.ax.set_xlabel('Time [S]')
        self.ax.set_ylabel('TTS [S]')

class plot_appperf:
    def __init__(self, ax, params, lps):
        self.ax = ax

        # too lazy to figure out axhspan's object. fix this later
        self.update(params, lps)

    def update(self, params, lps):

        cfg = params['cfg']
        cur_t = params['cur']
        gxsec = params['gxsec']

        self.ax.cla() # this is a brute-force way to update

        self.ax.set_xlim([cur_t-gxsec, cur_t])
        self.ax.autoscale_view(scaley=True)
        self.ax.set_ylim(bottom=0)

        x = lps.getlistx()
        y = lps.getlisty()
        self.ax.plot(x,y, label='')

#        self.ax.legend(loc='lower left', prop={'size':9})
        self.ax.set_xlabel('Time [S]')
        self.ax.set_ylabel('App performance')

class plot_runtime: # mean, std
    def __init__(self, ax, params, pdata):
        self.ax = ax
        self.update(params, pdata)

    def update(self, params, pdata, ptype = 'temp'):
        cfg = params['cfg']
        cur_t = params['cur']
        gxsec = params['gxsec']

        self.ax.cla()
        self.ax.set_xlim([cur_t-gxsec, cur_t])
        self.ax.autoscale_view(scaley=True)

        x = pdata.getlistx()
        y = pdata.getlisty()
        e = pdata.getlisto()
        self.ax.plot(x,y, scaley=True,  label='')
        self.ax.errorbar(x,y,yerr=e, lw=.2,  label = '')

        # we need to update labels everytime because of cla()
        self.ax.set_xlabel('Time [S]')
        self.ax.set_ylabel('Runtime')
#        self.ax.legend(loc='lower left', prop={'size':9})
# ----------------------

class plot_rapl:
    def __init__(self, ax, params, ppkg, pmem, titlestr=''):
        self.ax = ax
        self.titlestr = titlestr
        # too lazy to figure out axhspan's object. fix this later
        self.update(params, ppkg, pmem)

    def update(self, params, ppkg, pmem):

        cfg = params['cfg']
        cur_t = params['cur']
        gxsec = params['gxsec']

        self.ax.cla() # this is a brute-force way to update

        self.ax.set_xlim([cur_t-gxsec, cur_t])
        self.ax.autoscale_view(scaley=True)

        #self.ax.axis([cur_t-gxsec, cur_t, cfg['pwrmin'], cfg['pwrmax']]) # [xmin,xmax,ymin,ymax]

        pkgid = 0
        for t in ppkg:
            x = t.getlistx()
            ycap = t.getlisto()
            #self.ax.plot(x,ycap, scaley=False, color='red', label='PKG%dlimit'%pkgid )
            self.ax.plot(x,ycap, color='red', label='PKG%dlimit'%pkgid )
            pkgid += 1

        pkgid = 0
        for t in ppkg:
            x = t.getlistx()
            y = t.getlisty()
            #self.ax.plot(x,y,scaley=False,color=params['pkgcolors'][pkgid], label='PKG%d'%pkgid)
            self.ax.plot(x,y,color=params['pkgcolors'][pkgid], label='PKG%d'%pkgid)
            pkgid += 1


        pkgid = 0
        for t in pmem:
            x = t.getlistx()
            y = t.getlisty()
            #self.ax.plot(x,y,scaley=False,color=params['pkgcolors'][pkgid], linestyle='--', label='PKG%ddram'%pkgid)
            self.ax.plot(x,y,color=params['pkgcolors'][pkgid], linestyle='--', label='PKG%ddram'%pkgid)
            pkgid += 1

        self.ax.legend(loc='lower left', prop={'size':9})
        self.ax.set_xlabel('Time [S]')
        self.ax.set_ylabel('Power [W]')
        if len(self.titlestr):
            self.ax.set_title("%s" % self.titlestr)



class plot_line_err: # used for temp and freq (mean+std)
    def __init__(self, ax, params, pdata, ptype = 'temp' ):
        self.ax = ax

        # unfortunately, I couldn't figure out how to update errorbar correctly
        self.update(params, pdata, ptype)
        
        
    def update(self, params, pdata, ptype = 'temp'):
        cfg = params['cfg']
        cur_t = params['cur']
        gxsec = params['gxsec']

        self.ax.cla() # this is a brute-force way to update. I don't know how to update errorbar correctly.
        if ptype == 'temp':
            self.ax.axis([cur_t-gxsec, cur_t, cfg['tempmin'], cfg['tempmax']]) # [xmin,xmax,ymin,ymax]
        elif ptype == 'freq':
            self.ax.axis([cur_t-gxsec, cur_t, cfg['freqmin'], cfg['freqmax']]) # [xmin,xmax,ymin,ymax]
            plt.axhspan(cfg["freqnorm"], cfg["freqmax"], facecolor='#eeeeee', alpha=0.5)
        else:
            self.ax.axis([cur_t-gxsec, cur_t, 0, 100]) # [xmin,xmax,ymin,ymax]

        pkgid = 0
        for t in pdata:
            x = t.getlistx()
            y = t.getlisty()
            e = t.getlisto()
            self.ax.plot(x,y,scaley=False,color=params['pkgcolors'][pkgid], label='PKG%d'%pkgid)
            self.ax.errorbar(x,y,yerr=e, lw=.2, color=params['pkgcolors'][pkgid], label = '')
            pkgid += 1

        # we need to update labels everytime because of cla()
        self.ax.set_xlabel('Time [S]')
        if ptype == 'temp':
            self.ax.set_ylabel('Core temperature [C]')
        elif ptype == 'freq':
            self.ax.set_ylabel('Frequency [GHz]')
        else:
            self.ax.set_ylabel('Unknown')
        self.ax.legend(loc='lower left', prop={'size':9})

# below are kind of examples
#


class plotline:
    def __init__(self, ax, x, y):
        self.ax = ax
        self.line, = ax.plot(x,y)

        self.ax.axhspan( 0.7, 1.0, facecolor='#eeeeee', alpha=1.0)
        
    def update(self, x, y):
        self.line.set_data(x, y)


        
class plotcolormap:
    def __init__(self, ax, X):
        self.ax = ax
        self.im = self.ax.imshow(X, cmap=cm.jet, interpolation='nearest')
        self.im.set_cmap('spectral')
        self.im.set_clim(0, 1.5)
        
        f = plt.gcf()
        f.colorbar(self.im)

    def update(self,X):
        self.im.set_array(X)

class plotbar:
    def __init__(self, ax, x, y):
        self.ax = ax
        self.rects = ax.bar(x, y)

    def update(self, y):
        for r, h in zip(self.rects, y):
            r.set_height(h)

class ploterrorbar:
    def __init__(self, ax, x, y, e):
        self.ax = ax
        l, (b, t), v = ax.errorbar(x, y, e)
        self.line = l
        self.bottom = b
        self.top = t
        self.vert = v

    def update(self, x, y, e):
        # XXX: this is a bit brute-force
        # I couldn't figure out how to update vert
        self.ax.cla()
        self.ax.errorbar(x, y, e)

class plottext:
    def ypos(self,i):
        return 1.0 - 0.05*i

    def __init__(self, ax, n):
        self.ax = ax

        dir=os.path.abspath(os.path.dirname(sys.argv[0]))

        ax.axis([0,1,0,1])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on=True

        ax.plot( [ 0.1, 0.2], [0.96, 0.96], color='blue',  linewidth=2 )
        ax.plot( [ 0.1, 0.2], [0.91, 0.91], color='green', linewidth=2 )
        ax.plot( [ 0.1, 0.2], [0.86, 0.86], color='red',   linewidth=1 )

        self.text1 = ax.text( 0.3, self.ypos(2), '%d' % n )

        fn = get_sample_data("%s/coolr-logo-poweredby-48.png" % dir, asfileobj=False)
        arr = read_png(fn)
        imagebox = OffsetImage(arr, zoom=0.4)
        ab = AnnotationBbox(imagebox, (0, 0),
                            xybox=(.75, .12),
                            xycoords='data',
                            boxcoords="axes fraction",
                            pad=0.5)
        ax.add_artist(ab)

    def update(self, n):
        self.text1.set_text('%d' % n)
