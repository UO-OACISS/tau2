from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0
config = "hlrs"
inApp = "mg"
inExp = "parameter"
inTrial = "lu.A.4"

def load():
	print "loading data..."
	Utilities.setSession(config)
	trial1 = Utilities.getTrial(inApp, inExp, inTrial)
	result = TrialResult(trial1)
	print "...done."
	return result

def first(input):
	# get the iteration inclusive totals

	print "searching for iteration events (no classpath)..."
	events = ArrayList()
	for event in input.getEvents():
		#if event.find("Iteration") >= 0 and input.getEventGroupName(event).find("TAU_PHASE") < 0:
		if event.find("MPI_Send") >= 0 and event.find("message size") >= 0:
			events.add(event)
	print "...done."

	print "extracting phases..."
	extractor = ExtractEventOperation(input, events)
	extracted = extractor.processData().get(0)
	print "...done."

	return extracted

def second(extracted):
	print "getting stats..."
	dostats = BasicStatisticsOperation(extracted, False)
	stats = dostats.processData()
	print "...done."

	print "drawing charts..."
	for metric in stats.get(0).getMetrics():
		grapher = DrawMMMGraph(stats)
		metrics = HashSet()
		metrics.add(metric)
		grapher.set_metrics(metrics)
		grapher.setSortXAxis(True)
		grapher.setLogYAxis(True)
		grapher.setStripXName("MPI_Send\(\)   \[ <message size> = <")
		grapher.setTitle("NPB3.2.1 mg.A.4 - MPI_Send() Performance: " + metric)
		grapher.setSeriesType(DrawMMMGraph.TRIALNAME);
		grapher.setCategoryType(DrawMMMGraph.EVENTNAME)
		grapher.setValueType(AbstractResult.EXCLUSIVE)
		grapher.setXAxisLabel("Message Size")
		grapher.setYAxisLabel("Exclusive " + metric);
		# grapher.setLogYAxis(True)
		grapher.processData()
	print "...done."

	return

print "--------------- JPython test script start ------------"

loaded = load()
extracted = first(loaded)
second(extracted)

print "---------------- JPython test script end -------------"
