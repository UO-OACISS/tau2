from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
import math

tauData = ""
masterMeans = None
iterationPrefix = "Iteration"
nonMPI = "Computation"
MPI = "MPI"
kernNonMPI = "Kernel Computation"
kernMPI = "Kernel MPI"
init = "MPI_Init"
final = "MPI_Finalize"

def getParameters():
	global tauData
	global iterationPrefix
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
		iterationPrefix = tmp
		print "Iteration Prefix: " + iterationPrefix
	else:
		print "Iteration Prefix not specified... using", iterationPrefix

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def doLoadImbalance(trial, clusterID):
	# extract the non-callpath events from the trial
	#print "extracting non-callpath events",
	global masterMeans
	trial.setIgnoreWarnings(True)
	extractor = ExtractNonCallpathEventOperation(trial)
	extracted = extractor.processData().get(0)
	mainEventLong = extracted.getMainEvent()
	mainEvent = Utilities.shortenEventName(mainEventLong)
	#print "Main Event: ", mainEvent

	# compute the load imbalance
	#print "computing load imbalance",
	splitter = LoadImbalanceOperation(extracted)
	splitter.setPercentage(False)
	loadBalance = splitter.processData()
	conversion = 1.0 / 1000000.0

	thread = 0
	metric = trial.getTimeMetric()
	#event = LoadImbalanceOperation.KERNEL_COMPUTATION
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

	inclusive = masterMeans.getInclusive(0, mainEventLong, metric) * conversion
	print "%d\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t" % (clusterID, trial.getThreads().size(), inclusive, event, mean, max, min, stddev, ratio)
	clusterID = clusterID + 1
	return clusterID

def computeLoadBalance(trial, callpath, numphases):
	# extract the non-callpath events from the trial
	global masterMeans
	trial.setIgnoreWarnings(True)
	extracted = trial
	if callpath != True:
		extractor = ExtractNonCallpathEventOperation(trial)
		extracted = extractor.processData().get(0)
	mainEventLong = extracted.getMainEvent()
	mainEvent = extracted.getMainEvent()
	#print "Main Event: ", mainEvent

	# compute the load imbalance
	splitter = LoadImbalanceOperation(extracted)
	splitter.setPercentage(False)
	loadBalance = splitter.processData()
	conversion = 1.0 / 1000000.0
				
	thread = 0
	metric = trial.getTimeMetric()
	#event = LoadImbalanceOperation.KERNEL_COMPUTATION
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
	#print mean, max, min, stddev, ratio

	inclusive = masterMeans.getInclusive(0, mainEventLong, metric) * conversion

	if numphases < 100:
		print "%s\t %d\t %.2f\t %ls\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t" % (mainEvent, trial.getThreads().size(), inclusive, event, mean, max, min, stddev, ratio)

	return mean, max, min, stddev, inclusive

def myMax(a, b):
	if a > b:
		return a
	return b

def myMin(a, b):
	if a < b:
		return a
	return b

def main():
	global filename
	global iterationPrefix
	global masterMeans
	print "--------------- JPython test script start ------------"
	print "doing cluster test"
	# get the parameters
	getParameters()
	# load the data
	result = loadFile(tauData)
	result.setIgnoreWarnings(True)

	# set the metric, type we are interested in
	metric = result.getTimeMetric()
	type = result.EXCLUSIVE

	# extracting non-callpath events
	result.setIgnoreWarnings(True)
	extractor = ExtractNonCallpathEventOperation(result)
	extracted = extractor.processData().get(0)

	extracted.setIgnoreWarnings(True)
	print "Getting basic statistics..."
	statter = BasicStatisticsOperation(extracted)
	masterStats = statter.processData()
	masterMeans = masterStats.get(BasicStatisticsOperation.MEAN)
	
	# split communication and computation
	print "splitting communication and computation"
	splitter = SplitCommunicationComputationOperation(extracted)
	outputs = splitter.processData()
	computation = outputs.get(SplitCommunicationComputationOperation.COMPUTATION)
	communication = outputs.get(SplitCommunicationComputationOperation.COMMUNICATION)
	#computation = result

	# do some basic statistics first
	print "doing stats"
	stats = BasicStatisticsOperation(computation)
	means = stats.processData().get(BasicStatisticsOperation.MEAN)

	# then, using the stats, find the top X event names
	print "getting top X events"
	reducer = TopXEvents(means, metric, type, 10)
	reduced = reducer.processData().get(0)

	# then, extract those events from the actual data
	print "extracting events"
	tmpEvents = ArrayList(reduced.getEvents())
	reducer = ExtractEventOperation(computation, tmpEvents)
	reduced = reducer.processData().get(0)

	# cluster the data 
	print "clustering data"
	clusterer = DBSCANOperation(reduced, metric, type, 1.0)
	clusterResult = clusterer.processData()
	k = str(clusterResult.get(0).getThreads().size())
	clusters = ArrayList()
	print "Estimated value for k:", k
	if k > 0:
		clusterIDs = clusterResult.get(4)

		# split the trial into the clusters
		print "splitting clusters into", k, "trials"
		splitter = SplitTrialClusters(result, clusterResult)
		splitter.setIncludeNoisePoints(True)
		clusters = splitter.processData()
	else:
		clusters.put(result)

	clusterID = -1
	print "\nCluster\t Procs\t Incl.\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	clusterID = doLoadImbalance(result, clusterID)

	for trial in clusters:
		#print str(clusterID), trial.getMainEvent()
		clusterID = doLoadImbalance(trial, clusterID)

	#loopPrefix = "Iteration "
	#loopNames = set()
	#for event in trial.getEvents():
		#if event.find(loopPrefix) > -1:
			#loopNames.add(event)
			
	clusterID = 0
	for trial in clusters:
		#print "Getting basic statistics..."
		#statter = BasicStatisticsOperation(trial)
		#masterStats = statter.processData()
		#masterMeans = masterStats.get(BasicStatisticsOperation.MEAN)

		print "\n\nSplitting phases in cluster", clusterID
		splitter = SplitTrialPhasesOperation(trial, iterationPrefix)
		phases = splitter.processData()
		#print phases.size()
		totalMean = 0.0
		totalInclusive = 0.0
		avgMax = 0.0
		avgMin = 1.0
		totalMax = 0.0
		totalMin = 0.0
		totalStddev = 0.0
		totalRatio = 0.0

		print "LoopID\t\t Procs\t Incl.\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
		for phase in phases:
			#print "main event:", phase.getMainEvent()
			#for event in phase.getEvents():
			#print event
			mean, max, min, stddev, inclusive = computeLoadBalance(phase, True, phases.size())
			totalMean = totalMean + mean
			totalMax = totalMax + max
			totalMin = totalMin + min
			avgMax = myMax(avgMax, max)
			avgMin = myMin(avgMin, min)
			totalStddev = totalStddev + (stddev * stddev)
			totalInclusive = totalInclusive + inclusive

		avgMean = totalMean / phases.size()
		avgMax = totalMax / phases.size()
		avgMin = totalMin / phases.size()
		avgStddev = math.sqrt(totalStddev / phases.size())
		avgRatio = avgMean / avgMax
		avgInclusive = totalInclusive / phases.size()

		#event = LoadImbalanceOperation.KERNEL_COMPUTATION
		event = LoadImbalanceOperation.COMPUTATION
		print "%s\t\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t" % ("Average", trial.getThreads().size(), avgInclusive, event, avgMean, avgMax, avgMin, avgStddev, avgRatio)
		clusterID = clusterID + 1
	
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()

