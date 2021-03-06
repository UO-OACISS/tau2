from edu.uoregon.tau.perfdmf import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.common import AlphanumComparator
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *
from java.lang import *

True = 1
False = 0

tauData = "phases.ppk"
prefix = "Iteration"

def loadFile():
	# load the trial
	input = TrialResult(Utilities.getCurrentTrial())
	return input

def drawGraph(results):
	grapher = DrawGraph(results)
	grapher.setLogYAxis(False)
	grapher.setShowZero(True)
	grapher.setTitle("Graph of Phases")
	grapher.setSeriesType(DrawGraph.METRICNAME)

	# scale all values by 1,000,000
	grapher.setUnits(DrawGraph.SECONDS)

	# set the x axis category
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setXAxisLabel("Iteration Count")

	# set the y axis value
	grapher.setValueType(AbstractResult.INCLUSIVE)
	grapher.setYAxisLabel("Inclusive value")

	# sort the event names. We have to do this here, 
	# because it is not implemented in DrawGraph?
	comparator = AlphanumComparator()
	tmpSet = TreeSet(comparator)
	for event in results.getEvents():
		tmpSet.add(event)
	grapher.setEvents(tmpSet)

	grapher.processData()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	global tauData
	global prefix

	# read parameters and parse input data
	inputData = loadFile()

	# get rid of callpath events
	inputData.setIgnoreWarnings(True)
	extractor = ExtractNonCallpathEventOperation(inputData)
	extracted = extractor.processData().get(0)

	# extract the events of interest
	events = ArrayList()
	for event in extracted.getEvents():
		if event.startswith(prefix):
			events.add(event)
	extractor = ExtractEventOperation(extracted, events)
	extracted = extractor.processData().get(0)

	# draw the graph
	drawGraph(extracted)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
