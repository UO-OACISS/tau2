from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
from java.lang import *

tauData = ""

###################################################################

def getParameters():
	global parameterMap
	global tauData
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	#for key in keys:
		#print key, parameterMap.get(key)
	tmp = parameterMap.get("tauData")
	if tmp != None:
		tauData = tmp
		print "Performance data: " + tauData
	else:
		print "TAU profile data path not specified... using current directory of profile.x.x.x files."

def loadFile(fileName):
	global gprof
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("gprof.out"):
		input = DataSourceResult(DataSourceResult.GPROF, files, True)
		gprof = True
	elif fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

###################################################################

def computeLoadBalance(trial, callpath):
	# extract the non-callpath events from the trial
	trial.setIgnoreWarnings(True)
	extracted = None
	if callpath:
		extracted = trial
	else:
		extractor = ExtractNonCallpathEventOperation(trial)
		extracted = extractor.processData().get(0)
	mainEvent = extracted.getMainEvent()
	#print "Main Event: ", mainEvent

	# compute the load imbalance
	splitter = LoadImbalanceOperation(extracted)
	loadBalance = splitter.processData()
				
	thread = 0
	metric = trial.getTimeMetric()
	event = LoadImbalanceOperation.KERNEL_COMPUTATION

	means = loadBalance.get(LoadImbalanceOperation.MEAN)
	maxs = loadBalance.get(LoadImbalanceOperation.MAX)
	mins = loadBalance.get(LoadImbalanceOperation.MIN)
	stddevs = loadBalance.get(LoadImbalanceOperation.STDDEV)
	ratios = loadBalance.get(LoadImbalanceOperation.LOAD_BALANCE)

	mean = means.getExclusive(thread, event, metric)
	max = maxs.getExclusive(thread, event, metric)
	min = mins.getExclusive(thread, event, metric)
	stddev = stddevs.getExclusive(thread, event, metric)
	ratio = ratios.getExclusive(thread, event, metric)

	if callpath:
		#print "%s\t %d\t %ls\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (mainEvent, trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)
		print "%s\t %d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (mainEvent, trial.getThreads().size(), event, mean*100, 100, 100, 100, 100)
	else:
		print "%d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)


###################################################################

def main():
	global tauData

	print "--------------- JPython test script start ------------"
	print "--- Looking for load imbalances --- "

	# get the parameters
	getParameters()

	# load the data
	trial = loadFile(tauData)

	print "Procs\t Type\t\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	computeLoadBalance(trial, False)

	print

	# splitter = SplitTrialPhasesOperation(trial, "loop")
	splitter = SplitTrialPhasesOperation(trial, "Iteration")
	phases = splitter.processData()

	print "LoopID\t Procs\t Type\t\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	for phase in phases:
		computeLoadBalance(phase, True)

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
