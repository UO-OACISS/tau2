from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def loadDB(app, exp, trial):
	trial = Utilities.getTrial(app, exp, trial)
	input = TrialMeanResult(trial)
	return input

def loadFromDB():
	Utilities.setSession("advection_test")
	metadata = HashMap()
	conjoin = " and "
	# here are the conditions for the selection
	metadata.put("Application"," = 'advection'")
	metadata.put("Experiment"," = 'edison_final_thread'")
	metadata.put("dataFile"," = 'fishtank'")
	metadata.put("trialname"," like '%Med_8_4%'")
	trials = Utilities.getTrialsFromMetadata(metadata, conjoin)
	inputs = ArrayList()
	for trial in trials:
		#inputs.add(TrialMeanResult(trial, None, "tRunAlgorithm", False))
		inputs.add(TrialMeanResult(trial))
	return inputs

def drawGraph(results):
	# change this to P_WALL_CLOCK_TIME as necessary
	metric = "TIME"
	grapher = DrawGraph(results)
	metrics = HashSet()
	metrics.add(metric)
	grapher.setMetrics(metrics)
	grapher.setLogYAxis(False)
	grapher.setShowZero(True)
	grapher.setTitle("Graph of Multiple Trials: " + metric)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setUnits(DrawGraph.SECONDS)

	# for processor count X axis
	#grapher.setCategoryType(DrawGraph.PROCESSORCOUNT)

	# for trial name X axis
	grapher.setCategoryType(DrawGraph.TRIALNAME)

	# for metadata field on X axis
	#grapher.setCategoryType(DrawGraph.METADATA)
	#grapher.setMetadataField("pid")

	grapher.setXAxisLabel("Trial")
	grapher.setValueType(AbstractResult.EXCLUSIVE)
	grapher.setYAxisLabel("Exclusive " + metric + " (seconds)")
	grapher.processData()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	inputs = loadFromDB()

	# extract the event of interest
	events = ArrayList()
	# change this to zoneLimitedGradient(PileOfScalars) as necessary
	events.add("tRunAlgorithm")
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()

	drawGraph(extracted)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
