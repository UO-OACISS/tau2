from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def loadFromDB(app):
	Utilities.setSession("PowerExample")
	metadata = HashMap()
	conjoin = " and "
	# here are the conditions for the selection
	metadata.put("Application"," = '" + app + "'")
	inputs = ArrayList()
	events = ArrayList()
	trials = Utilities.getTrialsFromMetadata(metadata, conjoin)
	for trial in trials:
		inputs.add(TrialMeanResult(trial))
	return inputs

def drawGraph(results,app):
	# change this to P_WALL_CLOCK_TIME as necessary
	metric = "TIME"
	grapher = DrawGraph(results)
	metrics = HashSet()
	metrics.add(metric)
	grapher.setMetrics(metrics)
	events = HashSet()
	events.add("Computation")
	events.add("MPI")
	grapher.setEvents(events)
	#grapher.setLogYAxis(True)
	grapher.setShowZero(True)
	grapher.setTitle("Inclusive Execution Time Per Power Cap for " + app)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setUnits(DrawGraph.SECONDS)
	grapher.setCategoryType(DrawGraph.METADATA)
	grapher.setMetadataField("Experiment")
	grapher.setXAxisLabel("Power Cap")
	grapher.setValueType(AbstractResult.INCLUSIVE)
	grapher.setYAxisLabel("Inclusive " + metric + " (seconds)")
	grapher.processData()

def aggregateMPI(inputs):
	outputs = ArrayList()
	for i in inputs:
		splitter = SplitCommunicationComputationOperation(i)
		outputs = splitter.processData()
		computation = outputs.get(SplitCommunicationComputationOperation.COMPUTATION)
		communication = outputs.get(SplitCommunicationComputationOperation.COMMUNICATION)
		metric = "TIME"
		newEvent = "Computation"
		i.putCalls(0,newEvent,0)
		i.putSubroutines(0,newEvent,0)
		i.putInclusive(0,newEvent,metric,0)
		i.putExclusive(0,newEvent,metric,0)
		for e in computation.getEvents():
			i.putCalls(0,newEvent,i.getCalls(0,e)+i.getCalls(0,newEvent))
			i.putSubroutines(0,newEvent,i.getSubroutines(0,e)+i.getCalls(0,newEvent))
			i.putInclusive(0,newEvent,metric,i.getInclusive(0,e,metric)+i.getInclusive(0,newEvent,metric))
			i.putExclusive(0,newEvent,metric,i.getExclusive(0,e,metric)+i.getExclusive(0,newEvent,metric))
		newEvent = "MPI"
		i.putCalls(0,newEvent,0)
		i.putSubroutines(0,newEvent,0)
		i.putInclusive(0,newEvent,metric,0)
		i.putExclusive(0,newEvent,metric,0)
		for e in communication.getEvents():
			i.putCalls(0,newEvent,i.getCalls(0,e)+i.getCalls(0,newEvent))
			i.putSubroutines(0,newEvent,i.getSubroutines(0,e)+i.getCalls(0,newEvent))
			i.putInclusive(0,newEvent,metric,i.getInclusive(0,e,metric)+i.getInclusive(0,newEvent,metric))
			i.putExclusive(0,newEvent,metric,i.getExclusive(0,e,metric)+i.getExclusive(0,newEvent,metric))
	return inputs

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	for app in ["amg2013","CoMD-mpi","lulesh"]:
		inputs = loadFromDB(app)
		inputs = aggregateMPI(inputs)
		drawGraph(inputs,app)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
