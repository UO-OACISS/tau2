from client import ScriptFacade
from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import Utilities
from glue import TrialMeanResult
from glue import DifferenceOperation
from glue import AbstractResult
from edu.uoregon.tau.perfdmf import Trial

def glue(pe):
	print "doing glue test"
	Utilities.setSession("peri_test")
	trial1 = Utilities.GetTrial("GTC_s_PAPI", "VN XT3", "004")
	trial2 = Utilities.GetTrial("GTC_s_PAPI", "VN XT3", "008")
	result1 = TrialMeanResult(trial1)
	result2 = TrialMeanResult(trial2)
	operation = DifferenceOperation(result1)
	operation.addInput(result2)
	outputs = operation.processData()
	diffs = outputs.get(0);

	for thread in diffs.getThreads():
		for event in diffs.getEvents():
			for metric in diffs.getMetrics():
				print thread , event , metric
				# print diffs.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE)

print "--------------- JPython test script start ------------"

pe = ScriptFacade()
glue(pe)

# pe.exit()

print "---------------- JPython test script end -------------"
