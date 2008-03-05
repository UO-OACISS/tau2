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

def glue():
	print "doing phase test for gtc on jaguar"
	# load the trial
	Utilities.setSession("perfdmf.demo")
	trial1 = Utilities.getTrial("gtc_bench", "Jaguar Compiler Options", "fastsse")
	result1 = TrialResult(trial1)

	# get the iteration inclusive totals

	events = ArrayList()
	for event in result1.getEvents():
		#if event.find("Iteration") >= 0 and result1.getEventGroupName(event).find("TAU_PHASE") < 0:
		if event.find("Iteration") >= 0 and event.find("=>") < 0:
			events.add(event)

	extractor = ExtractEventOperation(result1, events)
	extracted = extractor.processData().get(0)

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

	print "extracted phases..."

	# get the Statistics
	dostats = BasicStatisticsOperation(extracted, False)
	stats = dostats.processData()

	print "got stats..."

	"""for metric in stats.get(0).getMetrics():
		grapher = DrawMMMGraph(stats)
		metrics = HashSet()
		metrics.add(metric)
		grapher.set_metrics(metrics)
		grapher.setTitle("GTC Phase Breakdown: " + metric)
		grapher.setSeriesType(DrawMMMGraph.TRIALNAME);
		grapher.setCategoryType(DrawMMMGraph.EVENTNAME)
		grapher.setValueType(AbstractResult.INCLUSIVE)
		grapher.setXAxisLabel("Iteration")
		grapher.setYAxisLabel("Inclusive " + metric);
		# grapher.setLogYAxis(True)
		grapher.processData()

	return"""

	# graph the significant events in the iteration

	subsetevents = ArrayList()
	subsetevents.add("CHARGEI")
	subsetevents.add("PUSHI")
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

		# get the Statistics
		dostats = BasicStatisticsOperation(extracted, False)
		stats = dostats.processData()

		print "got stats..."

		for metric in stats.get(0).getMetrics():
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
