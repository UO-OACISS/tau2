from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
from java.lang import *
import math

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
	mainEvent = Utilities.shortenEventName(extracted.getMainEvent())
	#print "Main Event: ", mainEvent
	if mainEvent == "Iteration 1001":
		return 0, 0, 0, 0

	# compute the load imbalance
	splitter = LoadImbalanceOperation(extracted)
	#splitter.setPercentage(False)
	loadBalance = splitter.processData()
				
	thread = 0
	metric = trial.getTimeMetric()
	#event = LoadImbalanceOperation.KERNEL_COMPUTATION
	event = LoadImbalanceOperation.COMPUTATION

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
	#if callpath:
		#for event in extracted.getEvents():
			#print event
		#print mean, max, min, stddev, ratio

	if callpath:
		print "%s\t %d\t %ls\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (mainEvent, trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)
		#print "%s\t %d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (mainEvent, trial.getThreads().size(), event, mean*100, 100, 100, 100, 100)
	else:
		print "%d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)

	return mean, max, min, stddev


###################################################################

def myMax(a, b):
	if a > b:
		return a
	return b

def myMin(a, b):
	if a < b:
		return a
	return b

def main():
	global tauData

	print "--------------- JPython test script start ------------"
	print "--- Looking for load imbalances --- "

	# get the parameters
	getParameters()

	# load the data
	trial = loadFile(tauData)

	print "Procs\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	computeLoadBalance(trial, False)

	print

	#for phaseName in ["int main", "Iteration"]:
	for phaseName in ["Iteration"]:
		# splitter = SplitTrialPhasesOperation(trial, "loop")
		# splitter = SplitTrialPhasesOperation(trial, "Iteration")
		splitter = SplitTrialPhasesOperation(trial, phaseName)
		phases = splitter.processData()
		totalMean = 0.0
		avgMax = 0.0
		avgMin = 1.0
		totalStddev = 0.0
		totalRatio = 0.0

		print "LoopID\t\t Procs\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
		for phase in phases:
			mean, max, min, stddev = computeLoadBalance(phase, True)
			if mean == max == min == stddev == 0:
				continue
			totalMean = totalMean + mean
			avgMax = myMax(avgMax, max)
			avgMin = myMin(avgMin, min)
			totalStddev = totalStddev + (stddev * stddev)

		avgMean = totalMean / phases.size()
		avgStddev = math.sqrt(totalStddev / phases.size())
		avgRatio = avgMean / avgMax

		#event = LoadImbalanceOperation.KERNEL_COMPUTATION
		event = LoadImbalanceOperation.COMPUTATION
		print "%s\t\t %d\t %ls\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % ("Average", trial.getThreads().size(), event, avgMean*100, avgMax*100, avgMin*100, avgStddev*100, avgRatio*100)
	
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
