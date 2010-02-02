from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
from java.lang import *
import math

tauData = ""
masterMeans = None
iterationPrefix = "Iteration"
vectorT_i = []

###################################################################

def getParameters():
	global parameterMap
	global tauData
	global iterationPrefix
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

	tmp = parameterMap.get("prefix")
	if tmp != None:
		iterationPrefix = tmp
		print "Iteration Prefix: " + iterationPrefix
	else:
		print "Iteration Prefix not specified... using", iterationPrefix

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
	elif fileName.endswith("xml"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

###################################################################

def computeLoadBalance(trial, callpath, numPhases):
	global masterMeans
	global iterationPrefix
	global vectorT_i
	# extract the non-callpath events from the trial
	trial.setIgnoreWarnings(True)
	extracted = trial
	#if not callpath:
		#extractor = ExtractNonCallpathEventOperation(trial)
		#extracted = extractor.processData().get(0)
	mainEventLong = extracted.getMainEvent()
	mainEvent = mainEventLong
	if not callpath:
		mainEvent = Utilities.shortenEventName(mainEventLong)
	#print "Main Event: ", mainEvent

	# compute the load imbalance
	splitter = LoadImbalanceOperation(extracted)
	splitter.setPercentage(False)
	loadBalance = splitter.processData()
				
	thread = 0
	metric = trial.getTimeMetric()
	conversion = 1.0 / 1000000.0
	event = LoadImbalanceOperation.COMPUTATION

	means = loadBalance.get(LoadImbalanceOperation.MEAN)
	maxs = loadBalance.get(LoadImbalanceOperation.MAX)
	mins = loadBalance.get(LoadImbalanceOperation.MIN)
	stddevs = loadBalance.get(LoadImbalanceOperation.STDDEV)
	ratios = loadBalance.get(LoadImbalanceOperation.LOAD_BALANCE)

	mean = means.getExclusive(thread, event, metric) * conversion
	max = maxs.getExclusive(thread, event, metric) * conversion
	min = mins.getExclusive(thread, event, metric) * conversion
	stddev = stddevs.getExclusive(thread, event, metric) * conversion
	ratio = ratios.getExclusive(thread, event, metric) 
	#if callpath:
		#for event in extracted.getEvents():
			#print event
		#print mean, max, min, stddev, ratio

	#for event in extracted.getEvents():
		#print event, means.getInclusive(thread, mainEvent, metric) * conversion

	#inclusive = means.getInclusive(thread, mainEventLong, metric) * conversion
	inclusive = masterMeans.getInclusive(0, mainEventLong, metric) * conversion
	threads = trial.getThreads().size()
	if callpath:
		if numPhases < 100:
			print "%s\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f" % (mainEvent, threads, inclusive, event, mean, max, min, stddev, ratio)
		#print "%s\t %d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (mainEvent, trial.getThreads().size(), event, mean*100, 100, 100, 100, 100)
	else:
		print "%d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f" % (threads, inclusive, event, mean, max, min, stddev, ratio)

	splits = loadBalance.get(LoadImbalanceOperation.COMPUTATION_SPLITS)
	for thread in splits.getThreads():
		vectorT_i[thread] = splits.getExclusive(thread, event, metric) * conversion

	return mean, max, min, stddev, inclusive


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
	global masterMeans
	global iterationPrefix
	global vectorT_i

	print "--------------- JPython test script start ------------"
	print "--- Looking for load imbalances --- "

	# get the parameters
	getParameters()

	# load the data
	trial = loadFile(tauData)
	trial.setIgnoreWarnings(True)
	print "Getting basic statistics..."
	statter = BasicStatisticsOperation(trial)
	masterStats = statter.processData()
	masterMeans = masterStats.get(BasicStatisticsOperation.MEAN)

	totalVectorT_i = []
	for thread in trial.getThreads():
		totalVectorT_i.append(0)
		vectorT_i.append(0)

	print "Procs\t Incl.\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	computeLoadBalance(trial, False, 1)

	print

	splitter = SplitTrialPhasesOperation(trial, iterationPrefix)
	phases = splitter.processData()
	totalMean = 0.0
	totalInclusive = 0.0
	avgMax = 0.0
	avgMin = 1.0
	totalMax = 0.0
	totalMin = 1.0
	totalStddev = 0.0
	totalRatio = 0.0

	print "LoopID\t\t Procs\t Incl.\t  Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	print "----------------------------------------------------------------------------------------"
	for phase in phases:
		mean, max, min, stddev, inclusive = computeLoadBalance(phase, True, phases.size())
		if mean == max == min == stddev == 0:
			continue
		totalMean = totalMean + mean
		avgMax = myMax(avgMax, max)
		avgMin = myMin(avgMin, min)
		totalMax = totalMax + max
		totalMin = totalMin + min
		totalStddev = totalStddev + (stddev * stddev)
		totalInclusive = totalInclusive + inclusive
		for index, item in enumerate(vectorT_i):
			totalVectorT_i[index] = totalVectorT_i[index] + item

	avgMean = totalMean / phases.size()
	avgMax = totalMax / phases.size()
	avgMin = totalMin / phases.size()
	avgStddev = math.sqrt(totalStddev / phases.size())
	avgRatio = avgMean / avgMax
	avgInclusive = totalInclusive / phases.size()

	event = LoadImbalanceOperation.COMPUTATION
	#print "%s\t\t %d\t %ls\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % ("Average", trial.getThreads().size(), event, avgMean*100, avgMax*100, avgMin*100, avgStddev*100, avgRatio*100)
	print "----------------------------------------------------------------------------------------"
	print "%s\t\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f" % ("Totals", trial.getThreads().size(), totalInclusive, event, totalMean, totalMax, totalMin, math.sqrt(totalStddev), totalMean / totalMax)
	print "%s\t\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f" % ("Average", trial.getThreads().size(), avgInclusive, event, avgMean, avgMax, avgMin, avgStddev, avgRatio)

	maxT_i = 0
	for value in totalVectorT_i:
		maxT_i = myMax(maxT_i, value)

	print "\nT:\t\t", totalInclusive
	print "T ideal:\t", totalMax
	print "max(T_i):\t", maxT_i
	print "LB:\t\t", avgRatio
	print "microLB:\t", maxT_i / totalMax
	print "Transfer:\t", totalMax / totalInclusive
	print "n\t\t", avgRatio * (maxT_i / totalMax) * (totalMax / totalInclusive) * 1.0, "\n"
	
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
