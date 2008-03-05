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
	trial = Utilities.getTrial("gtc_bench", "jaguar.longrun2", "64")
	result = TrialMeanResult(trial)

	events = ArrayList()
	for event in result.getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(result, events)
	extracted = extractor.processData().get(0)
	grapher = DrawGraph(extracted)
	# metrics = HashSet()
	# metrics.add("PAPI_L2_TCM")
	# grapher.set_metrics(metrics)
	grapher.setCategoryType(DrawGraph.EVENTNAME)
	grapher.setValueType(AbstractResult.INCLUSIVE)
	# grapher.setLogYAxis(True)
	grapher.processData()


print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
