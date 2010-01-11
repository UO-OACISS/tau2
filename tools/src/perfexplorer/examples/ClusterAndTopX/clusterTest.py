from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *

def getParameters():
	global tauData
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
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
	if fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	elif fileName.endswith("gprof"):
		input = DataSourceResult(DataSourceResult.GPROF, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def main():
	global filename
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
	
	# split communication and computation
	splitter = SplitCommunicationComputationOperation(result)
	outputs = splitter.processData()
	computation = outputs.get(SplitCommunicationComputationOperation.COMPUTATION)
	communication = outputs.get(SplitCommunicationComputationOperation.COMMUNICATION)
	#computation = result

	# do some basic statistics first
	stats = BasicStatisticsOperation(computation)
	means = stats.processData().get(BasicStatisticsOperation.MEAN)

	# then, using the stats, find the top X event names
	reducer = TopXEvents(means, metric, type, 20)
	reduced = reducer.processData().get(0)

	# then, extract those events from the actual data
	tmpEvents = ArrayList(reduced.getEvents())
	reducer = ExtractEventOperation(computation, tmpEvents)
	reduced = reducer.processData().get(0)

	# cluster the data 
	clusterer = DBSCANOperation(reduced, metric, type, 1.0)
	clusterResult = clusterer.processData()
	print "Estimated value for k:", str(clusterResult.get(0).getThreads().size())
	clusterIDs = clusterResult.get(4)

	# split the trial into the clusters
	splitter = SplitTrialClusters(result, clusterResult)
	clusters = splitter.processData()

	functions = "function-list.txt"
	gprof = False
	threshold = 10
	myFile = open(functions, 'w')
	for input in clusters:

		# extract the non-callpath data
		print "Extracting non-callpath data..."
		input.setIgnoreWarnings(True)
		extractor = ExtractNonCallpathEventOperation(input)
		extracted = extractor.processData().get(0)

		# extract computation code (remove MPI)
		myEvents = ArrayList()
		print "Filtering out MPI calls..."
		#print "And functions called less than 1000 times..."
		for event in extracted.getEvents():
			if not event.startswith("MPI_"):
				#if extracted.getCalls(extracted.getThreads().first(), event) > 999:
				myEvents.add(event)
		extractor = ExtractEventOperation(extracted, myEvents)
		extracted = extractor.processData().get(0)

		# generate statistics
		print "Generating stats..."
		doStats = BasicStatisticsOperation(extracted, False)
		mean = doStats.processData().get(BasicStatisticsOperation.MEAN)

		# get the top X events
		print "Extracting top events..."
		mean.setIgnoreWarnings(True)
		topper = TopXEvents(mean, mean.getTimeMetric(), AbstractResult.EXCLUSIVE, threshold)
		topped = topper.processData().get(0)

		# put the top X events names in a file
		for event in topped.getEvents():
			shortEvent = event
			# fix gprof names
			if gprof:
				shortEvent = shortEvent.upper()
				if shortEvent.startswith("__MODULE"):
					shortEvent = shortEvent.replace("__MODULE","MODULE")
					shortEvent = shortEvent.replace("_NMOD_","::")
			# fix TAU names
			else:
				shortEvent = Utilities.shortenEventName(event)
			percentage = topped.getExclusive(0,event,topped.getTimeMetric()) / mean.getInclusive(0,mean.getMainEvent(),mean.getTimeMetric()) * 100.0
			calls = topped.getCalls(0,event)
			if calls == 0.0:
				print "%00.2f%%\t %d\t %0.5f%%\t %s" % (percentage, calls, 0.0, shortEvent)
			else:
				print "%00.2f%%\t %d\t %0.5f%%\t %s" % (percentage, calls, percentage / float(calls), shortEvent)

			#print "%00.2f%%\t %s" % (topped.getExclusive(0,event,topped.getTimeMetric()) / mean.getInclusive(0,mean.getMainEvent(),mean.getTimeMetric()) * 100.0, event)
			myFile.write(shortEvent + "\n")
	myFile.close()

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()

