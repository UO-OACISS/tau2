from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *

True = 1

def glue():
	print "doing long run test for ocracoke"
	# load the trial
	Utilities.setSession("perfdmf_test")
	trial1 = Utilities.getTrial("gtc_bench", "ocracoke.longrun", "256p_5000ts_100micell")
	#Utilities.setSession("perfdmf.demo")
	#trial1 = Utilities.getTrial("gtc_bench", "Ocracoke longrun", "256p_5000ts_100micell")
	result1 = TrialMeanResult(trial1)

	events = ArrayList()
	for event in result1.getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(result1, events)
	extracted = extractor.processData().get(0)

	for metric in extracted.getMetrics():
		if metric == "GET_TIME_OF_DAY":
			grapher = DrawGraph(extracted)
			metrics = HashSet()
			metrics.add(metric)
			grapher.set_metrics(metrics)
			grapher.setTitle(metric)
			grapher.setCategoryType(DrawGraph.EVENTNAME)
			# grapher.setTitle("GTC Phase Breakdown: " + metric)
			grapher.setTitle("GTC Phase Breakdown: P_WALL_CLOCK_TIME")
			grapher.setXAxisLabel("Iteration group (100 iterations each)")
			grapher.setYAxisLabel("Inclusive " + metric + " microseconds");
			grapher.setValueType(AbstractResult.INCLUSIVE)
			# grapher.setLogYAxis(True)
			grapher.processData()

print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
