from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def loadFromDB():
	Utilities.setSession("super_test")
	metadata = HashMap()
	conjoin = " and "
	# here are the conditions for the selection
	metadata.put("days"," = '5'")
	metadata.put("machine"," = 'eos'")
	metadata.put("model_size"," = '15km'")
	# get just the first trial
	metadata.put("processes"," = '1024'")
	trials = Utilities.getTrialsFromMetadata(metadata, conjoin)
	metrics = ArrayList()
	metrics.add("WALL_CLOCK_TIME")
	baseline = TrialMeanResult(trials.get(0), None, None, False)
	extractor = TopXEvents(baseline, metrics.get(0), AbstractResult.EXCLUSIVE, 10)
	topx = extractor.processData().get(0)
	inputs = ArrayList()
	#inputs.add(topx)
	events = ArrayList()
	# unfortunately, the events are returned as a set. :(
	events.addAll(topx.getEvents())
	# just the rest of the trials
	metadata.remove("processes")
	trials = Utilities.getTrialsFromMetadata(metadata, conjoin)
	for trial in trials:
		# get just one event
		inputs.add(TrialMeanResult(trial, metrics, events, False))
		# or get all events
		#inputs.add(TrialMeanResult(trial))
	return inputs

def drawGraph(results):
	# change this to P_WALL_CLOCK_TIME as necessary
	metric = "WALL_CLOCK_TIME"
	grapher = DrawGraph(results)
	metrics = HashSet()
	metrics.add(metric)
	grapher.setMetrics(metrics)
	grapher.setLogYAxis(True)
	grapher.setShowZero(True)
	grapher.setTitle("Graph of Multiple Trials: " + metric)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setUnits(DrawGraph.SECONDS)
	grapher.setCategoryType(DrawGraph.METADATA)
	grapher.setMetadataField("processes")
	grapher.setXAxisLabel("Processes")
	grapher.setValueType(AbstractResult.EXCLUSIVE)
	grapher.setYAxisLabel("Exclusive " + metric + " (seconds)")
	#grapher.setChartType(DrawGraph.STACKEDAREACHART)
	grapher.processData()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	inputs = loadFromDB()

	"""
	for i in inputs:
		print i.getName()
		for e in i.getEvents():
			print e
	"""

	drawGraph(inputs)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
