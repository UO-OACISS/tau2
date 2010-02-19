from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
from java.lang import *
import math

tauData = ""
masterMeans = None
masterMaxs = None
iterationPrefix = "Iteration"
vectorT_i = []
vectorT = []

###################################################################

def trunc(s,min_pos=0,max_pos=75,ellipsis=True):
    # Sentinel value -1 returned by String function rfind
    NOT_FOUND = -1
    # Error message for max smaller than min positional error
    ERR_MAXMIN = 'Minimum position cannot be greater than maximum position'
    
    # If the minimum position value is greater than max, throw an exception
    if max_pos < min_pos:
        raise ValueError(ERR_MAXMIN)
    # Change the ellipsis characters here if you want a true ellipsis
    if ellipsis:
        suffix = '...'
    else:
        suffix = ''
    # Case 1: Return string if it is shorter (or equal to) than the limit
    length = len(s)
    if length <= max_pos:
        return s + suffix
    else:
        # Case 2: Return it to nearest period if possible
        try:
            end = s.rindex('.',min_pos,max_pos)
        except ValueError:
            # Case 3: Return string to nearest space
            end = s.rfind(' ',min_pos,max_pos)
            if end == NOT_FOUND:
                end = max_pos
        return s[0:end] + suffix

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
	global masterMaxs
	global iterationPrefix
	global vectorT_i
	global vectorT
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
	#inclusive = masterMeans.getInclusive(0, mainEventLong, metric) * conversion
	inclusive = masterMaxs.getInclusive(0, mainEventLong, metric) * conversion
	threads = trial.getThreads().size()
	if mean < 0 or max < 0 or min < 0 or stddev < 0 or inclusive < 0:
		return 0, 0, 0, 0, 0, 0
	if callpath:
		if numPhases < 100:
			print "%s\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f\t %.4f" % (trunc(mainEvent, max_pos=15), threads, inclusive, event, mean, max, min, stddev, max/inclusive, ratio)
		#print "%s\t %d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (mainEvent, trial.getThreads().size(), event, mean*100, 100, 100, 100, 100)
	else:
		print "%d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f\t %.4f" % (threads, inclusive, event, mean, max, min, stddev, max/inclusive, ratio)

	splits = loadBalance.get(LoadImbalanceOperation.COMPUTATION_SPLITS)
	for thread in splits.getThreads():
		vectorT_i[thread] = splits.getExclusive(thread, event, metric) * conversion
		vectorT[thread] = extracted.getInclusive(thread, mainEventLong, metric) * conversion

	return mean, max, min, stddev, inclusive, max/inclusive


###################################################################

def myMax(a, b):
	if a > b:
		return a
	return b

def myMin(a, b):
	if a < b:
		return a
	return b

def processCluster(trial, result):
	global tauData
	global masterMeans
	global masterMaxs
	global iterationPrefix
	global vectorT_i
	global vectorT

	print "Getting basic statistics..."
	trial.setIgnoreWarnings(True)
	statter = BasicStatisticsOperation(trial)
	masterStats = statter.processData()
	masterMeans = masterStats.get(BasicStatisticsOperation.MEAN)
	masterMaxs = masterStats.get(BasicStatisticsOperation.MAX)

	totalVectorT_i = []
	totalVectorT = []
	vectorT_i = []
	vectorT = []

	for thread in result.getThreads():
		totalVectorT_i.append(0)
		vectorT_i.append(0)
		totalVectorT.append(0)
		vectorT.append(0)

	#print "Procs\t Incl.\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"
	#computeLoadBalance(trial, False, 1)

	print

	splitter = SplitTrialPhasesOperation(trial, iterationPrefix)
	phases = splitter.processData()
	totalMean = 0.0
	totalInclusive = 0.0
	totalCommEff = 0.0
	avgMax = 0.0
	avgMin = 1.0
	totalMax = 0.0
	totalMin = 1.0
	totalStddev = 0.0
	totalRatio = 0.0

	print "LoopID\t\t Procs\t Incl.\t  Type\t\t AVG\t MAX\t MIN\t STDEV\t CommEff AVG/MAX"
	print "------------------------------------------------------------------------------------------------"
	for phase in phases:
		mean, max, min, stddev, inclusive, commEff = computeLoadBalance(phase, True, phases.size())
		if mean == max == min == stddev == 0:
			continue
		totalMean = totalMean + mean
		avgMax = myMax(avgMax, max)
		avgMin = myMin(avgMin, min)
		totalMax = totalMax + max
		totalMin = totalMin + min
		totalStddev = totalStddev + (stddev * stddev)
		totalInclusive = totalInclusive + inclusive
		totalCommEff = totalCommEff + commEff
		for index, item in enumerate(vectorT_i):
			totalVectorT_i[index] = totalVectorT_i[index] + item
		for index, item in enumerate(vectorT):
			totalVectorT[index] = totalVectorT[index] + item

	avgMean = totalMean / phases.size()
	avgMax = totalMax / phases.size()
	avgMin = totalMin / phases.size()
	avgStddev = math.sqrt(totalStddev / phases.size())
	avgRatio = avgMean / avgMax
	avgInclusive = totalInclusive / phases.size()
	avgCommEff = totalCommEff / phases.size()

	maxT_i = 0
	T = 0
	maxEff = 0
	totalEff = 0
	totalT_i = 0
	for value in totalVectorT:
		T = myMax(T, value)
	for value in totalVectorT_i:
		maxT_i = myMax(maxT_i, value)
		maxEff = myMax(maxEff, value/T)
		totalEff = totalEff + value/T
		totalT_i = totalT_i + value
	commEff = maxEff
	avgEff = totalEff / len(trial.getThreads())
	LB = avgEff / maxEff
	avgT_i = totalT_i / len(trial.getThreads())

	event = LoadImbalanceOperation.COMPUTATION
	#print "%s\t\t %d\t %ls\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % ("Average", trial.getThreads().size(), event, avgMean*100, avgMax*100, avgMin*100, avgStddev*100, avgRatio*100)
	print "------------------------------------------------------------------------------------------------"
	print "%s\t\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % ("Totals", trial.getThreads().size(), totalInclusive, event, totalMean, totalMax, totalMin, math.sqrt(totalStddev), totalCommEff, totalMean / totalMax)
	print "%s\t\t %d\t %.2f\t %s\t %.2f\t %.2f\t %.2f\t %.2f\t %.4f\t %.4f" % ("Average", trial.getThreads().size(), avgInclusive, event, avgMean, avgMax, avgMin, avgStddev, avgCommEff, avgRatio)

	# the total time spent in the loop.  Essentially, for each
	# iteration of the loop, get the total time for each process.  Accumulate
	# that vector over the whole loop.  The process with the longest time spent
	# computing (aggregated over all iterations) is the T.
	print "\nT:\t\t", T  
	# the total time spent computing, collapsed.  Essentially, for each
	# iteration of the loop, get the computing time for each process.  Accumulate
	# that vector over the whole loop.  The process with the longest time spent
	# computing (aggregated over all iterations) is the max(T_i).  
	print "max(T_i):\t", maxT_i
	print "avg(T_i):\t", avgT_i
	print "maxEff:\t\t", maxEff
	print "CommEff:\t", commEff, "(should be same as maxEff)"
	# the load balance for the loop.  This is the sum of all efficiencies for
	# all processes, divided by the number of processes times the maxiumum
	# efficiency.  This can be (and is) simplified, by summing the mean
	# computing times, and dividing by the max computing times.
	print "avgEff:\t\t", avgEff
	print "LB:\t\t", LB

	# the total time spent computing in the loop, serialized.  Essentially, for each
	# iteration of the loop, get the max computing time in that loop.  Add
	# those together.  Because of overlapping iterations, this can be larger
	# than the actual time in the loop.  If there were
	# no time spent in communication, this is how long the loop should take.
	print "T ideal:\t", totalMax
	# the micro load balance is the process with the highest computation time
	# divided by the ideal total loop execution time.
	print "microLB:\t", maxT_i / totalMax
	# the transfer term is the total time spent in the ideal loop divided by
	# the actual time spent in the loop.
	print "Transfer:\t", totalMax / T
	# finally, compute the efficiency.  == LB * microLB * Transfer * IPC
	print "n:\t\t", LB * (maxT_i / totalMax) * (totalMax / T) * 1.0, "\n"
	

def main():
	global tauData
	global masterMeans
	global masterMaxs
	global iterationPrefix
	global vectorT_i
	global vectorT

	print "--------------- JPython test script start ------------"
	print "--- Looking for load imbalances --- "

	# get the parameters
	getParameters()

	# load the data
	result = loadFile(tauData)
	result.setIgnoreWarnings(True)

######################################################################
# split into clusters
######################################################################

	# set the metric, type we are interested in
	metric = result.getTimeMetric()
	type = result.EXCLUSIVE

	# extract non-callpath
	extractor = ExtractNonCallpathEventOperation(result)
	extracted = extractor.processData().get(0)
	extracted.setIgnoreWarnings(True)

	# split communication and computation
	print "splitting communication and computation"
	splitter = SplitCommunicationComputationOperation(extracted)
	outputs = splitter.processData()
	computation = outputs.get(SplitCommunicationComputationOperation.COMPUTATION)

	# do some basic statistics first
	print "doing stats"
	simplestats = BasicStatisticsOperation(computation)
	simplemeans = simplestats.processData().get(BasicStatisticsOperation.MEAN)

	# get top 10 events
	print "getting top X events"
	reducer = TopXEvents(simplemeans, metric, type, 10)
	reduced = reducer.processData().get(0)
	print "extracting events"
	tmpEvents = ArrayList(reduced.getEvents())
	reducer = ExtractEventOperation(computation, tmpEvents)
	reduced = reducer.processData().get(0)

	# cluster
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

######################################################################
# done with clustering
######################################################################

	clusterID = 0
	for trial in clusters:
		processCluster(trial, result)

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
