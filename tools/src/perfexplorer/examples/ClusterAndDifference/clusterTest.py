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
	elif fileName.endswith("xml"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	elif fileName.endswith("gprof"):
		input = DataSourceResult(DataSourceResult.GPROF, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def diffs(baseline, comparison):
	baseline.setIgnoreWarnings(True)
	comparison.setIgnoreWarnings(True)
	extractor = ExtractNonCallpathEventOperation(baseline)
	extractor.addInput(comparison)
	extracted = extractor.processData()
	baseline = extracted.get(0)
	comparison = extracted.get(1)

	print "Baseline: ", baselineName
	print "Comparison: ", comparisonName

	# get the stats
	statMakerBaseline = BasicStatisticsOperation(baseline)
	baselineMeans = statMakerBaseline.processData().get(BasicStatisticsOperation.MEAN)
	baselineMeans.setIgnoreWarnings(True)
	statMakerComparison = BasicStatisticsOperation(comparison)
	comparisonMeans = statMakerComparison.processData().get(BasicStatisticsOperation.MEAN)
	comparisonMeans.setIgnoreWarnings(True)

	diff = DifferenceOperation(baselineMeans)
	diff.addInput(comparisonMeans)
	outputs = diff.processData()
	diffs = outputs.get(0)

	for type in AbstractResult.EXCLUSIVE, AbstractResult.INCLUSIVE, AbstractResult.CALLS:

		max = 10
		if type == AbstractResult.EXCLUSIVE:
			print "\nExclusive:\n"
		elif type == AbstractResult.INCLUSIVE:
			print "\nInclusive:\n"
			max = 20
		else:
			print "\nNumber of Calls:\n"

		# get the top 10?
		topXmaker = TopXEvents(diffs, baseline.getTimeMetric(), type, max)
		top10 = topXmaker.processData().get(0)

		print "B_Time  C_Time  D_Time  %_Diff  Event"
		print "------  ------  ------  ------  ------"
		for thread in top10.getThreads():
			for event in top10.getEvents():
				for metric in top10.getMetrics():
					baselineVal = baselineMeans.getDataPoint(thread, event, metric, type)
					comparisonVal = comparisonMeans.getDataPoint(thread, event, metric, type)
					diff = top10.getDataPoint(thread, event, metric, type)
					if type != AbstractResult.CALLS:
						baselineVal = baselineVal * 0.000001
						comparisonVal = comparisonVal * 0.000001
						diff = diff * 0.000001
					if baselineVal > comparisonVal:
						diff = diff * -1.0
					print "%.2f\t%.2f\t%.2f\t%.2f\t%s" % (baselineVal, comparisonVal, diff, (diff/baselineVal)*100.0, event)

	return

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
	
	# extract non-callpath events
	extractor = ExtractNonCallpathEventOperation(result)
	result = extractor.processData().get(0)

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
	baseline = clusters.get(0)
	for index in range(1,clusters.size()-1):
		# get the difference between clusters
		comparison = clusters.get(index)
		diff(baseline, comparison)
		


	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()

