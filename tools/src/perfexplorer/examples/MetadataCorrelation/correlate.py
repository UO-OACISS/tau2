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
	elif fileName.endswith("xml"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	elif fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PACKED, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def dumpData(inputData, metadata):
	global tauData
	metric = inputData.getTimeMetric()
	outfile = open(tauData+".csv", "w")
	outfile.write("pid, se_timestep, adv, se_btr_vel, equation_of_state, MPI_Wait(), nCellsSolve, nCells, nEdges, totalLevelCells, tauTotalLevelEdgeTop\n")
	for thread in inputData.getThreads():
		outfile.write(str(thread) + ", ")
		outfile.write(str(inputData.getExclusive(thread, "se timestep", metric)) + ", ")
		outfile.write(str(inputData.getExclusive(thread, "adv", metric)) + ", ")
		outfile.write(str(inputData.getExclusive(thread, "se btr vel", metric)) + ", ")
		outfile.write(str(inputData.getExclusive(thread, "equation of state", metric)) + ", ")
		outfile.write(str(inputData.getExclusive(thread, "MPI_Wait()", metric)) + ", ")
		outfile.write(str(metadata.getExclusive(thread, "nCellsSolve", "METADATA")) + ", ")
		outfile.write(str(metadata.getExclusive(thread, "nCells", "METADATA")) + ", ")
		outfile.write(str(metadata.getExclusive(thread, "nEdges", "METADATA")) + ", ")
		outfile.write(str(metadata.getExclusive(thread, "totalLevelCells", "METADATA")) + ", ")
		outfile.write(str(metadata.getExclusive(thread, "tauTotalLevelEdgeTop", "METADATA")) + " ")
		outfile.write("\n")
	outfile.close()

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
