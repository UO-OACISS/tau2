from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.common import AlphanumComparator
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.rules import *
from java.util import *
from java.lang import *

True = 1
False = 0

tauData = "tauprofile.xml"
ruleFile = "Causes.drl"

def getParameters():
	global tauData
	global path
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	path = parameterMap.get("path")
	tmp = parameterMap.get("tauData")
	if tmp != None:
		tauData = tmp
		print "Performance data: " + tauData
	else:
		print "TAU profile data path not specified... using current directory of profile.x.x.x files."

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("gz"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	elif "xml" in fileName:
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	elif fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PACKED, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def dumpData(inputData, metadata):
	global tauData
	metric = inputData.getTimeMetric()
	mpiwait = open(tauData+".mpiwait", "w")
	mpiwait2 = open(tauData+".mpiwaitcalls", "w")
	computation = open(tauData+".computation", "w")
	ncells = open(tauData+".ncells", "w")
	ncellssolve = open(tauData+".ncellssolve", "w")
	nodeid = open(tauData+".nodeid", "w")
	nNeededList = open(tauData+".nNeededList", "w")
	nOwnedList = open(tauData+".nOwnedList", "w")
	neighbors = open(tauData+".neighbors", "w")
	recvbytes = open(tauData+".recvbytes", "w")
	nodes=[]
	for thread in inputData.getThreads():
		bytes = inputData.getUsereventNumevents(thread, "Message size received from all nodes") * inputData.getUsereventMean(thread, "Message size received from all nodes")
		recvbytes.write(str(bytes) + "\n")
		mpi = inputData.getExclusive(thread, "MPI_Wait()", metric)
		comp = inputData.getInclusive(thread, "se timestep", metric)
		mpiwait.write(str(mpi) + "\n")
		computation.write(str(comp - mpi) + "\n")
		mpi = inputData.getCalls(thread, "MPI_Wait()")
		mpiwait2.write(str(mpi) + "\n")
		tmp = metadata.getExclusive(thread, "nCells", "METADATA")
		if tmp == 0.0:
			tmp = metadata.getExclusive(0, "nCells", "METADATA")
		ncells.write(str(tmp) + "\n")
		tmp = metadata.getExclusive(thread, "nCellsSolve", "METADATA")
		if tmp == 0.0:
			tmp = metadata.getExclusive(0, "nCellsSolve", "METADATA")
		ncellssolve.write(str(tmp) + "\n")
		tmp = metadata.getExclusive(thread, "nNeededList", "METADATA")
		if tmp == 0.0:
			tmp = metadata.getExclusive(0, "nNeededList", "METADATA")
		nNeededList.write(str(tmp) + "\n")
		tmp = metadata.getExclusive(thread, "nOwnedList", "METADATA")
		if tmp == 0.0:
			tmp = metadata.getExclusive(0, "nOwnedList", "METADATA")
		nOwnedList.write(str(tmp) + "\n")
		tmp = metadata.getExclusive(thread, "neighbors", "METADATA")
		if tmp == 0.0:
			tmp = metadata.getExclusive(0, "neighbors", "METADATA")
		neighbors.write(str(tmp) + "\n")
		tmp = metadata.getNameValue(thread, "Hostname")
		if tmp == None:
			tmp = metadata.getNameValue(0, "Hostname")
		if tmp not in nodes:
			nodes.append(tmp)
		nodeid.write(str(nodes.index(tmp)) + "\n")
		"""
		outfile.write(str(inputData.getExclusive(thread, "se timestep", metric)) + " ")
		outfile.write(str(inputData.getExclusive(thread, "adv", metric)) + " ")
		outfile.write(str(inputData.getExclusive(thread, "se btr vel", metric)) + " ")
		outfile.write(str(inputData.getExclusive(thread, "equation of state", metric)) + " ")
		outfile.write(str(metadata.getExclusive(thread, "nCellsSolve", "METADATA")) + " ")
		outfile.write(str(metadata.getExclusive(thread, "nEdges", "METADATA")) + " ")
		outfile.write(str(metadata.getExclusive(thread, "totalLevelCells", "METADATA")) + " ")
		outfile.write(str(metadata.getExclusive(thread, "tauTotalLevelEdgeTop", "METADATA")) + " ")
		"""
	recvbytes.close()
	mpiwait.close()
	mpiwait2.close()
	ncells.close()
	ncellssolve.close()
	computation.close()
	nodeid.close()
	nNeededList.close()
	nOwnedList.close()
	neighbors.close()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	global tauData
	global path

	# read parameters and parse input data
	getParameters()
	inputData = loadFile(tauData)

	ruleHarness = RuleHarness.useGlobalRules(path + ruleFile)

	inputData.setIgnoreWarnings(True)

	# get the metadata
	metadata = TrialThreadMetadata(inputData)
	#metadata.setIgnoreWarnings(True)

    # get the flat profile
	extractor = ExtractNonCallpathEventOperation(inputData)
	extracted = extractor.processData().get(0)

	# do some basic statistics first
	stats = BasicStatisticsOperation(extracted)
	means = stats.processData().get(BasicStatisticsOperation.MEAN)

	# then, using the stats, find the top X event names
	metric = extracted.getTimeMetric()
	type = extracted.EXCLUSIVE
	reducer = TopXEvents(means, metric, type, 5)
	reduced = reducer.processData().get(0)

	# then, extract those events from the actual data
	tmpEvents = ArrayList(reduced.getEvents())
	reducer = ExtractEventOperation(extracted, tmpEvents)
	reduced = reducer.processData().get(0)

	# correlate
	corr = CorrelateEventsWithMetadata(reduced, metadata)
	mycorr = corr.processData().get(0)

	# process the rules
	RuleHarness.getInstance().processRules()

	dumpData(inputData, metadata)

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
