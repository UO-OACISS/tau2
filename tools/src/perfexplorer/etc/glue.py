from client import ScriptFacade
from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import Utilities
from glue import TrialResult
from glue import BasicStatisticsOperation
from edu.uoregon.tau.perfdmf import Trial

def glue():
	print "doing glue test"
	Utilities.setSession("peri_gtc")
	trial = Utilities.getTrial("GTC", "ocracoke-O2", "64")
	trial2 = Utilities.getTrial("GTC", "ocracoke-O2", "128")
	trial3 = Utilities.getTrial("GTC", "ocracoke-O2", "256")
	result = TrialResult(trial)
	result2 = TrialResult(trial2)
	result3 = TrialResult(trial3)
	operation = BasicStatisticsOperation(result)
	operation.addInput(result2)
	operation.addInput(result3)
	outputs = operation.processData()
	total = outputs.get(0);
	mean = outputs.get(1);
	variance = outputs.get(2);
	stdev = outputs.get(3);

	for thread in total.getThreads():
		for event in total.getEvents():
			for metric in total.getMetrics():
				print thread , event , metric
				# print mean.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE)

print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
