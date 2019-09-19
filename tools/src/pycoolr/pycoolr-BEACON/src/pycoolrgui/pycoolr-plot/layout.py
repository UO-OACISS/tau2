#!/usr/bin/env python

import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

class layoutclass:
    def __init__(self, row=2, col=4):
        self.row = row
        self.col = col
        self.idx = 1
        #self.idx = 2

        #self.figure = fig
        self.fig = Figure(figsize=(15, 8), dpi=100)
        self.root = Tk.Tk()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        #rax = plt.axes([0.02, 0.8, 0.08, 0.1])
        #check = CheckButtons(rax, ('Memory', 'MPI_T_PVAR', 'NODE_POWER'), (False, True, False))

        #check.on_clicked(handlerCB)

    def getax(self):
        ax = plt.subplot(self.row, self.col, self.idx)
        self.idx += 1
        return ax

    
    def distrplots(self, fig):
        ax = fig.add_subplot(self.row,self.col,self.idx)
        self.idx += 1
        return ax
   

