from client import ScriptFacade
from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import MergeTrialsOperation
from glue import ExtractPhasesOperation
from glue import ExtractEventOperation
from glue import CorrelationOperation
from glue import Utilities
from glue import TrialMeanResult
from glue import AbstractResult
from glue import CorrelationResult
from glue import DrawGraph
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1

def glue():
	print "doing long run test"
	# load the trial
	Utilities.setSession("perfdmf_test")
	trial1 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.first")
	result1 = TrialMeanResult(trial1)

	events = ArrayList()
	for event in result1.getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(result1, events)
	extracted = extractor.processData().get(0)

	grapher = DrawGraph(extracted)
	metrics = HashSet()
	metrics.add("PAPI_L1_TCM")
	grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()

	grapher = DrawGraph(extracted)
	metrics = HashSet()
	metrics.add("P_WALL_CLOCK_TIME")
	grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()

	trial2 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.second")
	result2 = TrialMeanResult(trial2)

	events = ArrayList()
	for event in result2.getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(result2, events)
	extracted = extractor.processData().get(0)

	grapher = DrawGraph(extracted)
	metrics = HashSet()
	metrics.add("PAPI_FP_INS")
	grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()

	grapher = DrawGraph(extracted)
	metrics = HashSet()
	metrics.add("PAPI_L1_TCA")
	grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()

	trial3 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.third")
	result3 = TrialMeanResult(trial3)

	events = ArrayList()
	for event in result3.getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(result3, events)
	extracted = extractor.processData().get(0)

	grapher = DrawGraph(extracted)
	metrics = HashSet()
	metrics.add("PAPI_L2_TCM")
	grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()

	grapher = DrawGraph(extracted)
	metrics = HashSet()
	metrics.add("PAPI_TOT_INS")
	grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()


print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
