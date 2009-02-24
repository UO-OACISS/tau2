from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.glue import *
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
	trial2 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.second")
	result2 = TrialMeanResult(trial2)
	trial3 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.third")
	result3 = TrialMeanResult(trial3)

	# merge the trials together - they have different metrics
	merger = MergeTrialsOperation(result1)
	merger.addInput(trial2)
	# merger.addInput(trial3)
	merged = merger.processData()

	# extract the interval events
	reducer = ExtractPhasesOperation(merged.get(0), "Iteration")
	reduceds = reducer.processData()
	reduced = reduceds.get(0)

	for event in reduced.getEvents():
		for metric in reduced.getMetrics():
			for thread in reduced.getThreads():
				if event.find("measurement") >= 0:
					print metric, thread, reduced.getInclusive(thread, event, metric)

	# do the correlation
	correlation = CorrelationOperation(reduced)
	outputs = correlation.processData()
	result = outputs.get(0)

	# type = AbstractResult.INCLUSIVE;
	# for event in result.getEvents():
		# for metric in result.getMetrics():
			# for thread in result.getThreads():
				# if event.find("INCLUSIVE") >= 0:
					# print event, CorrelationResult.typeToString(thread), metric, ":", AbstractResult.typeToString(type), result.getDataPoint(thread, event, metric, type)

	events = ArrayList()
	for event in merged.get(0).getEvents():
		if event.find("Iteration") >= 0:
			events.add(event)

	extractor = ExtractEventOperation(merged.get(0), events)
	extracted = extractor.processData().get(0)

	for metric in extracted.getMetrics():
		grapher = DrawGraph(extracted)
		metrics = HashSet()
		metrics.add(metric)
		grapher.set_metrics(metrics)
		grapher.setCategoryType(DrawGraph.EVENTNAME)
		grapher.setValueType(AbstractResult.INCLUSIVE)
		grapher.setLogYAxis(True)
		grapher.processData()


print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
