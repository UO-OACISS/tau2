from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0

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

def diffs():

	baselineName = "phases32tg4.ppk"
	comparisonName = "phases32.ppk"

	# load the trials
	baseline = loadFile(baselineName)
	baseline.setIgnoreWarnings(True)
	comparison = loadFile(comparisonName)
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
	print "--------------- JPython test script start ------------"

	diffs()

	print "\n---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
