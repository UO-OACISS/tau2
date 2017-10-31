#
# File: pysamarc.py
# Authors: Jayanarayanan Sitaraman
# Last Modified: 1/24/06

# Standard Python modules

import sys
import os
import copy
import string
import types


#Extension modules

import numpy
# Try to import the mpi module
try:
	from mpi4py import MPI
	_parallel = True
	
except ImportError:
	_parallel = False
		

if _parallel:
	try:
		import samint
	except ImportError:
		print "Import Error: have to run with MPI"


class samarc:

	def  __init__(self,startfile):
		self.inpfile=startfile

		if not os.path.isfile(startfile):
			print 'Error: Could not find file %s' % startfile
			return None

		samint.samarcInit(self.inpfile)

	def getGlobalGridInfo(self):
		self.ngrids=samint.cvar.ngrids
		self.obgrids=[]
		
		
		for i in range(self.ngrids):
			self.obgrids.append(samint.getObgridsPy(i))

		self.gridParam=[]
		self.pilo=[]
		self.pihi=[]
		self.dx=[]
		self.xlo=[]
		self.maxLevel=0
		
		for i in range(self.ngrids):
			samint.setObgridDataPy(self.obgrids[i])
			self.pilo.append(samint.cvar.piloPy)
			self.pihi.append(samint.cvar.pihiPy)
			gridParam=[samint.cvar.global_idPy,
			samint.cvar.level_numPy,
			samint.cvar.level_idPy,
			samint.cvar.proc_idPy]
			self.xlo.append(samint.cvar.xloPy)
			self.dx.append(samint.cvar.dxPy)
			
			self.gridParam.append(gridParam)
			if samint.cvar.level_numPy > self.maxLevel:
				self.maxLevel=samint.cvar.level_numPy

		return self.gridParam,self.pilo,self.pihi,self.xlo,self.dx

	def getLocalPatchInfo(self):
		
		self.nlocal=samint.cvar.nlocal
		self.pdata=[]

		for i in range(self.nlocal):
			self.pdata.append(samint.getPdataPy(i))

		
		self.qParam=[]
		self.q=[]
		self.ibl=[]
		
		for i in range(self.nlocal):
			samint.setPatchDataPy(self.pdata[i])
			gid=samint.cvar.global_idPy
			if self.gridParam[gid][1] == self.maxLevel:
				self.q.append(samint.cvar.dataPy)
				self.ibl.append(samint.cvar.iblPy)
				qParam=[samint.cvar.global_idPy,
					samint.cvar.jdPy,
					samint.cvar.kdPy,
					samint.cvar.ldPy]
				self.qParam.append(qParam)

		return self.qParam,self.q,self.ibl

	def runStep(self,simtime,dt):
		samint.samarcStep(simtime,dt)

	def writePlotData(self,simtime,step):
		samint.samarcWritePlotData(simtime,step)
	
	def reGrid(self,simtime):
		samint.samarcRegrid(simtime)
	
	def finish(self):
		samint.samarcFinish()

	def update(self,simtime):
		self.time=simtime

	def runSubStep(self,simtime,dt,i,itnmax):
		self.time=simtime
