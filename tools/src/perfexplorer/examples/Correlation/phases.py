from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.common import AlphanumComparator
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *
from java.lang import *

True = 1
False = 0

tauData = "phases.ppk"
prefix = "iteration"

def getParameters():
	global tauData
	global prefix
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	tmp = parameterMap.get("tauData")
	if tmp != None:
		tauData = tmp
		print "Performance data: " + tauData
	else:
		print "TAU profile data path not specified... using current directory of profile.x.x.x files."

	tmp = parameterMap.get("prefix")
	if tmp != None:
		prefix = tmp
	else:
		print "Prefix not specified. Using default."
	print "Prefix: " + prefix


def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("gz"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	elif fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	global tauData
	global prefix

	# read parameters and parse input data
	getParameters()
	inputData = loadFile(tauData)

	# get rid of callpath events
	inputData.setIgnoreWarnings(True)
	extracted = inputData

	# extract the events of interest
	events = ArrayList()
	for event in extracted.getEvents():
		if event.startswith("[SUMMARY] mpas_ocn_tracer_advection_mono_tend "):
			events.add(event)
		if event.startswith("RK4-pronostic halo update => MPI_Wait()"):
			events.add(event)
	extractor = ExtractEventOperation(extracted, events)
	extracted = extractor.processData().get(0)

	corr = CorrelationOperation(extracted)
	mycorr = corr.processData().get(0)
	print mycorr.getCorrelation()

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
