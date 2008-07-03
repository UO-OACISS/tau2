from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import ExtractEventOperation
from glue import Utilities
from glue import BasicStatisticsOperation
from glue import DeriveMetricOperation
from glue import MergeTrialsOperation
from glue import TrialResult
from glue import AbstractResult
from glue import DrawMMMGraph
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0
config = "hlrs"
inApp = "lu"
inExp = "phases"
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
		if event.find("Iteration") >= 0 and event.find("=>") < 0 and event.find("**") < 0:
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
		grapher.setSortXAxis(False)
		grapher.setLogYAxis(False)
		grapher.setStripXName("Iteration")
		grapher.setTitle("NPB3.2.1 lu.A.4 - Iteration Performance: " + metric)
		grapher.setSeriesType(DrawMMMGraph.TRIALNAME);
		grapher.setCategoryType(DrawMMMGraph.EVENTNAME)
		grapher.setValueType(AbstractResult.INCLUSIVE)
		grapher.setXAxisLabel("Iteration Number")
		grapher.setYAxisLabel("Inclusive " + metric);
		# grapher.setLogYAxis(True)
		grapher.processData()
	print "...done."

	return

print "--------------- JPython test script start ------------"

loaded = load()
extracted = first(loaded)
second(extracted)

print "---------------- JPython test script end -------------"
