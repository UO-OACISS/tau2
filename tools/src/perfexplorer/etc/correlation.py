from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import Trial
from java.util import ArrayList

def glue():
	print "doing correlation test"
	Utilities.setSession("peri_gtc")
	trial = Utilities.getTrial("GTC", "ocracoke-O5", "2048")
	result = TrialResult(trial)
	events = ArrayList()
	events.add("CHARGEI [{chargei.F90} {1,12}]")
	events.add("SHIFTI [{shifti.F90} {1,12}]")
	reducer = ExtractEventOperation(result, events)
	reduced = reducer.processData();
	correlation = CorrelationOperation(reduced.get(0))
	outputs = correlation.processData()
	result = outputs.get(0)

	type = AbstractResult.EXCLUSIVE;
	for event in result.getEvents():
		for metric in result.getMetrics():
			for thread in result.getThreads():
				print event, CorrelationResult.typeToString(thread), metric, ":", AbstractResult.typeToString(type), result.getDataPoint(thread, event, metric, type)


print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
