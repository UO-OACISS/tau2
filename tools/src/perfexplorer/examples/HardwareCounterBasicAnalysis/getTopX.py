from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0
parameterMap = None
tauData = ""
threshold = 10
functions = "function-list.txt"
gprof = False

def getTopX(inTrial, threshold, timerType, metric=None, filterMPI=True):
	inTrial.setIgnoreWarnings(True)

	extracted = inTrial
	# extract computation code (remove MPI)
	if filterMPI:
		myEvents = ArrayList()
		print "Filtering out MPI calls..."
		for event in inTrial.getEvents():
			if not event.startswith("MPI_"):
				myEvents.add(event)
		extractor = ExtractEventOperation(inTrial, myEvents)
		extracted = extractor.processData().get(0)

	# put the top X events names in a list
	myEvents = ArrayList()

	# get the top X events
	print "Extracting top events..."
	extracted.setIgnoreWarnings(True)
	if metric is None:
		metric = extracted.getTimeMetric()
	topper = TopXEvents(extracted, metric, timerType, threshold) 
	topped = topper.processData().get(0)

	for event in topped.getEvents():
		shortEvent = Utilities.shortenEventName(event)
		tmp = extracted.getInclusive(0,extracted.getMainEvent(),metric)
		exclusivePercent = 0.0
		if tmp > 0:
			exclusivePercent = topped.getDataPoint(0,event,metric, timerType) / tmp * 100.0
		if (exclusivePercent > 1.0):
			print "%00.2f%%\t %d\t %s" % (exclusivePercent, extracted.getCalls(0,event), shortEvent)
			myEvents.add(event)
	return myEvents
