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
from java.lang import System

True = 1
False = 0

def load():
	print "loading data..."
	Utilities.setSession("perfdmf.demo")
	trial1 = Utilities.getTrial("gtc", "phases", "jaguar.64")
	result = TrialResult(trial1)
	print "...done."
	return result

def first(input):
	# get the iteration inclusive totals

	print "searching for iteration events (no classpath)..."
	events = ArrayList()
	for event in input.getEvents():
		#if event.find("Iteration") >= 0 and input.getEventGroupName(event).find("TAU_PHASE") < 0:
		if event.find("Iteration") >= 0 and event.find("=>") < 0:
			events.add(event)
	print "...done."

	print "extracting phases..."
	extractor = ExtractEventOperation(input, events)
	extracted = extractor.processData().get(0)
	print "...done."

	return extracted

# method to derive a metric, and merge it back into to the trial
def deriveStat(input, firstMetric, secondMetric, operation):
	derivor = DeriveMetricOperation(input, firstMetric, secondMetric, operation)
	derived = derivor.processData().get(0)
	merger = MergeTrialsOperation(input)
	merger.addInput(derived)
	input = merger.processData().get(0)
	return input

def second(extracted):
	print "deriving metrics..."
	extracted = deriveStat(extracted, "PAPI_L1_TCA", "PAPI_L1_TCM", DeriveMetricOperation.SUBTRACT)
	extracted = deriveStat(extracted, "(PAPI_L1_TCA-PAPI_L1_TCM)", "PAPI_L1_TCA", DeriveMetricOperation.DIVIDE)
	extracted = deriveStat(extracted, "PAPI_L1_TCM", "PAPI_L2_TCM", DeriveMetricOperation.SUBTRACT)
	extracted = deriveStat(extracted, "(PAPI_L1_TCM-PAPI_L2_TCM)", "PAPI_L1_TCM", DeriveMetricOperation.DIVIDE)
	extracted = deriveStat(extracted, "PAPI_FP_INS", "P_WALL_CLOCK_TIME", DeriveMetricOperation.DIVIDE)
	extracted = deriveStat(extracted, "PAPI_FP_INS", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	print "...done."

	print "getting stats..."
	dostats = BasicStatisticsOperation(extracted, False)
	stats = dostats.processData()
	print "...done."

	print "drawing charts..."
	for metric in stats.get(0).getMetrics():
		if metric != "(PAPI_L1_TCA-PAPI_L1_TCM)" and metric != "(PAPI_L1_TCM-PAPI_L2_TCM)" and metric != "PAPI_TOT_INS" and metric != "PAPI_L1_TCA" and metric != "PAPI_FP_INS":
			grapher = DrawMMMGraph(stats)
			metrics = HashSet()
			metrics.add(metric)
			grapher.set_metrics(metrics)
			if (metric == "(PAPI_FP_INS/P_WALL_CLOCK_TIME)"):
				grapher.setTitle("GTC Phase Breakdown: MFlop/s")
			elif (metric == "(PAPI_FP_INS/PAPI_TOT_INS)"):
				grapher.setTitle("GTC Phase Breakdown: FP fraction of instructions")
			elif (metric == "((PAPI_L1_TCA-PAPI_L1_TCM)/PAPI_L1_TCA)"):
				grapher.setTitle("GTC Phase Breakdown: L1 Cache Hit Rate")
			elif (metric == "((PAPI_L1_TCM-PAPI_L2_TCM)/PAPI_L1_TCM)"):
				grapher.setTitle("GTC Phase Breakdown: L2 Cache Hit Rate")
			elif (metric == "PAPI_L2_TCM"):
				grapher.setTitle("GTC Phase Breakdown: L2 Cache Misses")
			elif (metric == "PAPI_L1_TCM"):
				grapher.setTitle("GTC Phase Breakdown: L1 Cache Misses")
			elif (metric == "P_WALL_CLOCK_TIME"):
				grapher.setTitle("GTC Phase Breakdown: Total Wall Clock Time")
			else:
				grapher.setTitle("GTC Phase Breakdown: " + metric)
			grapher.setSeriesType(DrawMMMGraph.TRIALNAME);
			grapher.setCategoryType(DrawMMMGraph.EVENTNAME)
			grapher.setValueType(AbstractResult.INCLUSIVE)
			grapher.setXAxisLabel("Iteration")
			grapher.setYAxisLabel("Inclusive " + metric);
			# grapher.setLogYAxis(True)
			grapher.processData()
	print "...done."

	extracted = None
	stats = None
	System.gc()

	return

def third(input):
	# graph the significant events in the iteration

	subsetevents = ArrayList()
	subsetevents.add("CHARGEI")
	subsetevents.add("PUSHI")
	subsetevents.add("SHIFTI")

	for subsetevent in subsetevents:
		print "extracting callpath phases..."
		events = ArrayList()
		for event in input.getEvents():
			if event.find("Iteration") >= 0 and event.rfind(subsetevent) >= 0:
				events.add(event)

		extractor = ExtractEventOperation(input, events)
		extracted = extractor.processData().get(0)
		print "...done."

		# derive metrics

		print "deriving metrics..."
		extracted = deriveStat(extracted, "PAPI_L1_TCA", "PAPI_L1_TCM", DeriveMetricOperation.SUBTRACT)
		extracted = deriveStat(extracted, "(PAPI_L1_TCA-PAPI_L1_TCM)", "PAPI_L1_TCA", DeriveMetricOperation.DIVIDE)
		extracted = deriveStat(extracted, "PAPI_L1_TCM", "PAPI_L2_TCM", DeriveMetricOperation.SUBTRACT)
		extracted = deriveStat(extracted, "(PAPI_L1_TCM-PAPI_L2_TCM)", "PAPI_L1_TCM", DeriveMetricOperation.DIVIDE)
		extracted = deriveStat(extracted, "PAPI_FP_INS", "P_WALL_CLOCK_TIME", DeriveMetricOperation.DIVIDE)
		extracted = deriveStat(extracted, "PAPI_FP_INS", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
		print "...done."

		# get the Statistics
		print "getting stats..."
		dostats = BasicStatisticsOperation(extracted, False)
		stats = dostats.processData()
		print "...done."

		print "drawing charts..."
		for metric in stats.get(0).getMetrics():
			if metric == "((PAPI_L1_TCA-PAPI_L1_TCM)/PAPI_L1_TCA)" or metric == "((PAPI_L1_TCM-PAPI_L2_TCM)/PAPI_L1_TCM)":
				grapher = DrawMMMGraph(stats)
				metrics = HashSet()
				metrics.add(metric)
				grapher.set_metrics(metrics)
				if (metric == "(PAPI_FP_INS/P_WALL_CLOCK_TIME)"):
					grapher.setTitle(subsetevent + " Phase Breakdown: MFlop/s")
				elif (metric == "(PAPI_FP_INS/PAPI_TOT_INS)"):
					grapher.setTitle(subsetevent + " Phase Breakdown: FP fraction of instructions")
				elif (metric == "((PAPI_L1_TCA-PAPI_L1_TCM)/PAPI_L1_TCA)"):
					grapher.setTitle(subsetevent + " Phase Breakdown: L1 Cache Hit Rate")
				elif (metric == "((PAPI_L1_TCM-PAPI_L2_TCM)/PAPI_L1_TCM)"):
					grapher.setTitle(subsetevent + " Phase Breakdown: L2 Cache Hit Rate")
				elif (metric == "PAPI_L2_TCM"):
					grapher.setTitle(subsetevent + " Phase Breakdown: L2 Cache Misses")
				elif (metric == "PAPI_L1_TCM"):
					grapher.setTitle(subsetevent + " Phase Breakdown: L1 Cache Misses")
				elif (metric == "P_WALL_CLOCK_TIME"):
					grapher.setTitle(subsetevent + " Phase Breakdown: Total Wall Clock Time")
				else:
					grapher.setTitle(subsetevent + " Phase Breakdown: " + metric)
				#grapher.setTitle(subsetevent + ", " + metric)
				grapher.setSeriesType(DrawMMMGraph.TRIALNAME);
				grapher.setCategoryType(DrawMMMGraph.EVENTNAME)
				grapher.setValueType(AbstractResult.INCLUSIVE)
				# grapher.setLogYAxis(True)
				grapher.processData()
		print "...done."

	extracted = None
	stats = None
	System.gc()

	return

print "--------------- JPython test script start ------------"

loaded = load()
extracted = first(loaded)
second(extracted)
extracted = None
#third(loaded)
loaded = None
System.gc()

print "---------------- JPython test script end -------------"
