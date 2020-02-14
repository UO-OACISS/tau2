#!/usr/bin/env python3
#from mpi4py import MPI
import numpy as np
import adios2
#import os
#import glob
#from multiprocessing import Pool
#import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import operator
from operator import add
from matplotlib.font_manager import FontProperties

# Global variables
cpu_component_count = 9
cpu_components = ['Guest','I/O Wait', 'IRQ', 'Idle', 'Nice', 'Steal', 'System', 'User', 'soft IRQ']
previous_mean = {}
previous_count = {}
current_period = {}
period_values = {}
#mem_components = ['Memory Footprint (VmRSS) (KB)','Peak Memory Usage Resident Set Size (VmHWM) (KB)','meminfo:MemAvailable (MB)','meminfo:MemFree (MB)','meminfo:MemTotal (MB)']
mem_components = ['Memory Footprint (VmRSS) (KB)','Peak Memory Usage Resident Set Size (VmHWM) (KB)','program size (kB)','resident set size (kB)']
mem_components_short = ['VmRSS','VmHWM','program size','RSS']
io_components = ['io:cancelled_write_bytes', 'io:rchar', 'io:read_bytes', 'io:syscr', 'io:syscw', 'io:wchar', 'io:write_bytes']
io_components_short = ['cancelled_write_bytes', 'rchar', 'read_bytes', 'syscr', 'syscw', 'wchar', 'write_bytes']


def SetupArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instream", "-i", help="Name of the input stream", required=True)
    parser.add_argument("--outfile", "-o", help="Name of the output file", default="screen")
    parser.add_argument("--nompi", "-nompi", help="ADIOS was installed without MPI", action="store_true")
    parser.add_argument("--displaysec", "-dsec", help="Float representing gap between plot window refresh", default=0.2)
    args = parser.parse_args()

    args.displaysec = float(args.displaysec)
    args.nx = 1
    args.ny = 1
    args.nz = 1
    
    return args

def dumperiod_valuesars(vars_info):
    # print variables information
    for name, info in vars_info.items():
        print("variable_name: " + name)
        for key, value in info.items():
            print("\t" + key + ": " + value)
        print("\n")

def initialize_globals():
    global cpu_components
    global previous_mean
    global previous_count
    global current_period
    global period_values
    for c in cpu_components:
        previous_mean[c] = 0
        previous_count[c] = 0
        current_period[c] = 0
        period_values[c] = []

    for m in mem_components:
        previous_mean[m] = 0
        previous_count[m] = 0
        current_period[m] = 0
        period_values[m] = []

    for i in io_components:
        previous_mean[i] = 0
        previous_count[i] = 0
        current_period[i] = 0
        period_values[i] = []

def get_utilization(is_cpu, fr_step, vars_info, components, previous_mean, previous_count, current_period, period_values):
    for c in components:
        substr = c
        if is_cpu:
            substr = "cpu: "+c+" %"
        # Get the current mean value
        mean_var = substr + " / Mean"
        shape_str = vars_info[mean_var]["Shape"].split(',')
        shape = list(map(int,shape_str))
        mean_values = fr_step.read(mean_var)
        # Get the number of events
        count_var = substr + " / Num Events"
        shape2_str = vars_info[count_var]["Shape"].split(',')
        shape2 = list(map(int,shape2_str))
        count_values = fr_step.read(count_var)
        # Convert to MB if necessary
        if not is_cpu and "KB" in c.upper():
            mean_values[0] = mean_values[0] / 1000.0
        # Compute the total values seen 
        total_value = mean_values[0]*count_values[0]
        # What's the value from the current frame?
        if previous_count[c] < count_values[0]:
            current_period[c] = (total_value - (previous_mean[c] * previous_count[c])) / (count_values[0] - previous_count[c])
            previous_mean[c] = mean_values[0]
            previous_count[c] = count_values[0]
        #print(c,mean_values[0],count_values[0],total_value, current_period[c])
        period_values[c] = np.append(period_values[c], current_period[c])

def get_top5(fr_step, vars_info):
    num_ranks = 2
    num_threads = 8
    timer_means = {}
    timer_values = {}
    for name, info in vars_info.items():
        if ".TAU application" in name:
            continue
        if "Exclusive TIME" in name:
            shape_str = vars_info[name]["Shape"].split(',')
            shape = list(map(int,shape_str))
            mean_values = fr_step.read(name)
            shortname = name.replace(" / Exclusive TIME", "")
            timer_values[shortname] = []
            timer_values[shortname].append(mean_values[0])
            index = num_threads
            while index < shape[0]:
                timer_values[shortname].append(mean_values[index])
                index = index + num_threads
            timer_means[shortname] = np.sum(timer_values[shortname]) / num_ranks   
    limit = 0
    others = len(timer_means) - 5
    timer_values["other"] = [0] * num_ranks
    for key, value in sorted(timer_means.items(), key=lambda kv: kv[1]):
        limit = limit + 1
        if limit <= others:
            timer_values["other"] = list( map(add, timer_values["other"], timer_values[key]) )
            del timer_values[key]
    #print(timer_values)
    return timer_values

def plot_cpu_utilization(ax, x, fontsize):
    global cpu_components
    global previous_mean
    global previous_count
    global current_period
    global period_values
    ax.stackplot(x,period_values['Guest'],period_values['I/O Wait'],period_values['IRQ'],period_values['Idle'],period_values['Nice'],period_values['Steal'],period_values['System'],period_values['User'],period_values['soft IRQ'], labels=cpu_components)
    ax.legend(loc='lower left')

    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title("CPU Utilization", fontsize=fontsize)
    ax.set_xlabel("step", fontdict=fontdict)
    ax.set_ylabel("percent", fontdict=fontdict)
    ax.yaxis.tick_right()
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)

def plot_mem_utilization(ax, x, fontsize):
    global mem_components
    global previous_mean
    global previous_count
    global current_period
    global period_values
    for m,ms in zip(mem_components, mem_components_short):
        ax.plot(x,period_values[m],label=ms)
    ax.legend(loc='lower left')

    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title("Memory Utilization", fontsize=fontsize)
    ax.set_xlabel("step", fontdict=fontdict)
    ax.set_ylabel("MB", fontdict=fontdict)
    ax.yaxis.tick_right()
    ax.set_yscale('log')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)

def plot_io_utilization(ax, x, fontsize):
    global io_components
    global previous_mean
    global previous_count
    global current_period
    global period_values
    for m,ms in zip(io_components, io_components_short):
        ax.plot(x,period_values[m],label=ms)
    ax.legend(loc='lower left')

    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title("I/O", fontsize=fontsize)
    ax.set_xlabel("step", fontdict=fontdict)
    ax.set_ylabel(" ", fontdict=fontdict)
    ax.yaxis.tick_right()
    ax.set_yscale('log')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)

def plot_timers(ax, x, fontsize, top5):
    for key in top5:
        ax.bar(x,top5[key],label=((key[:30] + '..') if len(key) > 30 else key))
    ax.legend(loc='lower left')

    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title("Top 5 timers per rank", fontsize=fontsize)
    ax.set_xlabel("rank", fontdict=fontdict)
    ax.set_ylabel("Time (us)", fontdict=fontdict)
    ax.yaxis.tick_right()
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)

def plot_utilization(args, x, fontsize, step, top5):
    print("plotting", end='...', flush=True)
    fig = plt.figure(4, figsize=(8,8), constrained_layout=True)
    gs = gridspec.GridSpec(4, 1, figure=fig)
    cpu = fig.add_subplot(gs[0, 0])
    mem = fig.add_subplot(gs[1, 0])
    io = fig.add_subplot(gs[2, 0])
    timers = fig.add_subplot(gs[3, 0])

    plot_cpu_utilization(cpu, x, fontsize)
    plot_mem_utilization(mem, x, fontsize)
    plot_io_utilization(io, x, fontsize)
    plot_timers(timers, np.arange(2), fontsize, top5)
    plt.tick_params(axis='both', which='both', labelsize = fontsize/2)

    print("writing", end='...', flush=True)
    plt.ion()
    if (args.outfile == "screen"):
        plt.show()
        plt.pause(args.displaysec)
    else:
        imgfile = args.outfile+"_"+"{0:0>5}".format(step)+".png"
        fig.savefig(imgfile)

    plt.clf()
    print("done.") 

def process_file(args):
    fontsize=12
    filename = args.instream
    print ("Opening:", filename)
    if not args.nompi:
        fr = adios2.open(filename, "r", MPI.COMM_SELF, "adios.xml", "TAUProfileOutputPDF")
    else:
        fr = adios2.open(filename, "r", "adios.xml", "TAUProfileOutputPDF")
    initialize_globals()
    cur_step = 0
    for fr_step in fr:
        # track current step
        cur_step = fr_step.current_step()
        print(filename, "Step = ", cur_step)
        # inspect variables in current step
        vars_info = fr_step.available_variables()
        #dumperiod_valuesars(vars_info)
        get_utilization(True, fr_step, vars_info, cpu_components, previous_mean, previous_count, current_period, period_values)
        get_utilization(False, fr_step, vars_info, mem_components, previous_mean, previous_count, current_period, period_values)
        get_utilization(False, fr_step, vars_info, io_components, previous_mean, previous_count, current_period, period_values)
        top5 = get_top5(fr_step, vars_info)

        x=range(0,cur_step+1)
        plot_utilization(args, x, fontsize, cur_step, top5)

if __name__ == '__main__':
    args = SetupArgs()
    #print(args)
    process_file(args)

