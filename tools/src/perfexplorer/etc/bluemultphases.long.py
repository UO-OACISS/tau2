from client import ScriptFacade
from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import MergeTrialsOperation
from glue import ExtractPhasesOperation
from glue import ExtractEventOperation
from glue import CorrelationOperation
from glue import Utilities
from glue import TrialResult
from glue import AbstractResult
from glue import CorrelationResult
from glue import BasicStatisticsOperation
from glue import DrawMMMGraph
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1

def glue():
	print "doing long run test for ocracoke"
	# load the trial
	Utilities.setSession("perfdmf_test")
	# trial1 = Utilities.getTrial("gtc_bench", "superphases", "64")
	trial1 = Utilities.getTrial("gtc_bench", "ocracoke.longrun", "256p_5000ts_100micell")
	result1 = TrialResult(trial1)

	events = ArrayList()
	for event in result1.getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(result1, events)
	extracted = extractor.processData().get(0)

	print "extracted phases..."

	# get the Statistics
	dostats = BasicStatisticsOperation(extracted, False)
	stats = dostats.processData()

	print "got stats..."

	metrics = ArrayList()
	metrics.add("BGL_TIMERS")

	# for metric in extracted.getMetrics():
	for metric in metrics:
		grapher = DrawMMMGraph(stats)
		metrics = HashSet()
		metrics.add(metric)
		grapher.set_metrics(metrics)
		grapher.setTitle("BGL_TIMERS")
		grapher.setSeriesType(DrawGraph.TRIALNAME);
		grapher.setCategoryType(DrawGraph.EVENTNAME)
		grapher.setValueType(AbstractResult.INCLUSIVE)
		# grapher.setLogYAxis(True)
		grapher.processData()

print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
