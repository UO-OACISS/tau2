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
from glue import Provenance
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0

def glue():
	print "doing phase test for gtc on jaguar"

	operations = Provenance.getCurrent().getOperations()
	result1 = operations.get(0).getInputs().get(0)
	for operation in operations:
		print operation.getClass().getName()
		if operation.getClass().getName() == "glue.BasicStatisticsOperation":
			stats = operation.getOutputs()

	# graph the significant events in the iteration

	subsetevents = ArrayList()
	# subsetevents.add("CHARGEI")
	# subsetevents.add("PUSHI")
	subsetevents.add("SHIFTI")

	print "got data..."

	for subsetevent in subsetevents:
		events = ArrayList()
		for event in result1.getEvents():
			if event.find("Iteration") >= 0 and event.rfind(subsetevent) >= 0:
				events.add(event)

		extractor = ExtractEventOperation(result1, events)
		extracted = extractor.processData().get(0)

		print "extracted phases..."

		# derive metrics

		derivor = DeriveMetricOperation(extracted, "PAPI_L1_TCA", "PAPI_L1_TCM", DeriveMetricOperation.SUBTRACT)
		derived = derivor.processData().get(0)
		merger = MergeTrialsOperation(extracted)
		merger.addInput(derived)
		extracted = merger.processData().get(0)
		derivor = DeriveMetricOperation(extracted, "PAPI_L1_TCA-PAPI_L1_TCM", "PAPI_L1_TCA", DeriveMetricOperation.DIVIDE)
		derived = derivor.processData().get(0)
		merger = MergeTrialsOperation(extracted)
		merger.addInput(derived)
		extracted = merger.processData().get(0)
		derivor = DeriveMetricOperation(extracted, "PAPI_L1_TCM", "PAPI_L2_TCM", DeriveMetricOperation.SUBTRACT)
		derived = derivor.processData().get(0)
		merger = MergeTrialsOperation(extracted)
		merger.addInput(derived)
		extracted = merger.processData().get(0)
		derivor = DeriveMetricOperation(extracted, "PAPI_L1_TCM-PAPI_L2_TCM", "PAPI_L1_TCM", DeriveMetricOperation.DIVIDE)
		derived = derivor.processData().get(0)
		merger = MergeTrialsOperation(extracted)
		merger.addInput(derived)
		extracted = merger.processData().get(0)

		print "derived metrics..."

		# get the Statistics
		dostats = BasicStatisticsOperation(extracted, False)
		stats = dostats.processData()

		print "got stats..."

		for metric in stats.get(0).getMetrics():
			if metric != "PAPI_L1_TCA-PAPI_L1_TCM" and metric != "PAPI_L1_TCM-PAPI_L2_TCM":
				grapher = DrawMMMGraph(stats)
				metrics = HashSet()
				metrics.add(metric)
				grapher.set_metrics(metrics)
				grapher.setTitle(subsetevent + ", " + metric)
				grapher.setSeriesType(DrawMMMGraph.TRIALNAME);
				grapher.setCategoryType(DrawMMMGraph.EVENTNAME)
				grapher.setValueType(AbstractResult.INCLUSIVE)
				# grapher.setLogYAxis(True)
				grapher.processData()

	return

print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
