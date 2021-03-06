from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

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
	events = ArrayList()
	events.add("tRunAlgorithm")
	for trial in trials:
		# get just one event
		inputs.add(TrialMeanResult(trial, None, events, False))
		# or get all events
		#inputs.add(TrialMeanResult(trial))
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
	grapher.setCategoryType(DrawGraph.TRIALNAME)
	grapher.setXAxisLabel("Trial Name")
	grapher.setValueType(AbstractResult.EXCLUSIVE)
	grapher.setYAxisLabel("Exclusive " + metric + " (seconds)")
	grapher.processData()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	inputs = loadFromDB()

	drawGraph(inputs)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
