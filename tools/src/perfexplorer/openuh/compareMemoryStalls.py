import sys
from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TrialResult
from glue import BasicStatisticsOperation
from glue import AbstractResult
from glue import ExtractNonCallpathEventOperation
from glue import ExtractEventOperation
from glue import DeriveMetricOperation
from glue import ScaleMetricOperation
from glue import MergeTrialsOperation
from glue import DerivedMetrics
from glue import TopXEvents
from glue import MeanEventFact

###################################################################

True = 1
False = 0
mainEvent = ""
# set some metric names
#L1Hits = "PAPI_L1_TCA"
L1Hits = "PAPI_L1_DCH"
L1DataStalls = "L1_DATA_STALLS"
InstMiss = "IMISS"
L2Refs = "L2_REFERENCES"
L2DataRef = "L2_DATA_REFERENCES_L2_ALL"
L2Misses = "L2_MISSES"
L3Misses = "L3_MISSES"
TLBPenalty = "L2DTLB_MISSES"
EarCache = "DATA_EAR_CACHE_LAT128"
L2Hits = ""
L3Hits = ""
LocalMemoryHits = ""
RemoteMemoryHits = ""
RatioMemoryAccesses = ""
Total = ""
TotalRatio = ""

###################################################################

def deriveMetric(input, first, second, oper):
	# derive the metric
	derivor = DeriveMetricOperation(input, first, second, oper)
	derived = derivor.processData().get(0)
	newName = derived.getMetrics().toArray()[0]
	# merge new metric with the trial
	merger = MergeTrialsOperation(input)
	merger.addInput(derived)
	merged = merger.processData().get(0)
	#print "new metric: " + newName
	return merged, newName

def scaleMetric(input, metric, value, oper):
	# derive the metric
	scaler = ScaleMetricOperation(input, metric, value, oper)
	scaled = scaler.processData().get(0)
	newName = scaled.getMetrics().toArray()[0]
	# merge new metric with the trial
	merger = MergeTrialsOperation(input)
	merger.addInput(scaled)
	merged = merger.processData().get(0)
	#print "new metric: " + newName
	return merged, newName

###################################################################

def getMemoryModel(input):
# L1 hits: PAPI_L1_TCA * 1
# L2 hits: (PAPI_NATIVE_L2_data_references_L2_all – PAPI_NATIVE_L2_misses)*2
# L3 hits: (PAPI_NATIVE_L2_misses – PAPI_NATIVE_L3_misses)*10
# TLB penalty: PAPI_NATIVE_L2dtlb_misses * 30
# Local Memory Hits: (PAPI_NATIVE_L3_misses - DATA_EAR_CACHE_LAT128) *127
# Remote Memory Hits: PAPI_NATIVEDATA_EAR_CACHE_LAT128 *232
# Ratio of Remote memory accesses (PAPI_NATIVE_DATA_EAR_CACHE_LAT128 / PAPI_NATIVE_L3_misses)
# Ratio of Local Memory Accesses (PAPI_NATIVE_L3_misses - PAPI_NATIVEDATA_EAR_CACHE_LAT128) / PAPI_NATIVE_L3_misses 

	# set some metric names
	#L1Hits = "PAPI_L1_TCA"
	global L1Hits
	global L2Hits
	global L3Hits
	global L1DataStalls
	global InstMiss
	global L2Refs
	global L2DataRef
	global L2Misses
	global L3Misses
	global TLBPenalty
	global EarCache
	global LocalMemoryHits
	global RemoteMemoryHits
	global RatioMemoryAccesses
	global Total
	global TotalRatio

	global mainEvent

	# derive the L1 Hits term
	#print "\t", mainEvent, L1Hits, input.getInclusive(0, mainEvent, L1Hits)
	# print "\t", mainEvent, L2Refs, input.getInclusive(0, mainEvent, L2Refs)
	# input, L1Hits = deriveMetric(input, L1Hits, L2Refs, DeriveMetricOperation.SUBTRACT)
	# print "\t", mainEvent, L1Hits, input.getInclusive(0, mainEvent, L1Hits)

	# derive the L2 Hits term
	#print mainEvent, L2DataRef, input.getInclusive(0, mainEvent, L2DataRef)
	#print mainEvent, L2Misses, input.getInclusive(0, mainEvent, L2Misses)
	input, L2Hits = deriveMetric(input, L2DataRef, L2Misses, DeriveMetricOperation.SUBTRACT)
	#print mainEvent, L2Hits, input.getInclusive(0, mainEvent, L2Hits)
	input, L2Hits = scaleMetric(input, L2Hits, (2.0), ScaleMetricOperation.MULTIPLY)
	#print "\t", mainEvent, L2Hits, input.getInclusive(0, mainEvent, L2Hits)

	# derive the L3 Hits term
	#print mainEvent, L3Misses, input.getInclusive(0, mainEvent, L3Misses)
	input, L3Hits = deriveMetric(input, L2Misses, L3Misses, DeriveMetricOperation.SUBTRACT)
	#print mainEvent, L3Hits, input.getInclusive(0, mainEvent, L3Hits)
	input, L3Hits = scaleMetric(input, L3Hits, (10.0), ScaleMetricOperation.MULTIPLY)
	#print "\t", mainEvent, L3Hits, input.getInclusive(0, mainEvent, L3Hits)

	# derive the TLB Penalty term
	#print mainEvent, TLBPenalty, input.getInclusive(0, mainEvent, TLBPenalty)
	input, TLBPenalty = scaleMetric(input, TLBPenalty, (30.0), ScaleMetricOperation.MULTIPLY)
	#print "\t", mainEvent, TLBPenalty, input.getInclusive(0, mainEvent, TLBPenalty)

	# derive the Local Memory Hits:
	#print mainEvent, EarCache, input.getInclusive(0, mainEvent, EarCache)
	input, LocalMemoryHits = deriveMetric(input, L3Misses, EarCache, DeriveMetricOperation.SUBTRACT)
	#print mainEvent, LocalMemoryHits, input.getInclusive(0, mainEvent, LocalMemoryHits)
	input, LocalMemoryHits = scaleMetric(input, LocalMemoryHits, (127.0), ScaleMetricOperation.MULTIPLY)
	#print "\t", mainEvent, LocalMemoryHits, input.getInclusive(0, mainEvent, LocalMemoryHits)

	# Remote Memory Hits: PAPI_NATIVEDATA_EAR_CACHE_LAT128 *232
	input, RemoteMemoryHits = scaleMetric(input, EarCache, (232.0), ScaleMetricOperation.MULTIPLY)
	#print "\t", mainEvent, RemoteMemoryHits, input.getInclusive(0, mainEvent, RemoteMemoryHits)

	# sum them up
	input, Total = deriveMetric(input, L1Hits, L2Hits, DeriveMetricOperation.ADD)
	input, Total = deriveMetric(input, Total, L3Hits, DeriveMetricOperation.ADD)
	input, Total = deriveMetric(input, Total, TLBPenalty, DeriveMetricOperation.ADD)
	input, Total = deriveMetric(input, Total, LocalMemoryHits, DeriveMetricOperation.ADD)
	input, Total = deriveMetric(input, Total, RemoteMemoryHits, DeriveMetricOperation.ADD)
	#print "\t", mainEvent, Total, input.getInclusive(0, mainEvent, Total)
	#print mainEvent, "CPU_CYCLES", input.getInclusive(0, mainEvent, "CPU_CYCLES")
	input, TotalRatio = deriveMetric(input, Total, "CPU_CYCLES", DeriveMetricOperation.DIVIDE)
	#print "\t", mainEvent, TotalRatio, input.getInclusive(0, mainEvent, TotalRatio)

	# Ratio of Remote memory accesses (PAPI_NATIVE_DATA_EAR_CACHE_LAT128 / PAPI_NATIVE_L3_misses)
	input, RemoteMemoryHits = deriveMetric(input, EarCache, L3Misses, DeriveMetricOperation.DIVIDE)
	#print "\t", mainEvent, RemoteMemoryHits, input.getInclusive(0, mainEvent, RemoteMemoryHits)

	# Ratio of Local Memory Accesses (PAPI_NATIVE_L3_misses - PAPI_NATIVEDATA_EAR_CACHE_LAT128) / PAPI_NATIVE_L3_misses 
	input, RatioMemoryAccesses = deriveMetric(input, L3Misses, EarCache, DeriveMetricOperation.SUBTRACT)
	input, RatioMemoryAccesses = deriveMetric(input, RatioMemoryAccesses, L3Misses, DeriveMetricOperation.DIVIDE)
	#print "\t", mainEvent, RatioMemoryAccesses, input.getInclusive(0, mainEvent, RatioMemoryAccesses)

	# return the trial, and the new derived metric name
	return input, RatioMemoryAccesses

###################################################################

def main():
	global mainEvent
	global True
	global False
	global L1Hits
	global L2Hits
	global L3Hits
	global L1DataStalls
	global InstMiss
	global L2Refs
	global L2DataRef
	global L2Misses
	global L3Misses
	global TLBPenalty
	global EarCache
	global LocalMemoryHits
	global RemoteMemoryHits
	global RatioMemoryAccesses
	global Total
	global TotalRatio
	
	print "--------------- JPython test script start ------------"
	print "--- Calculating Memory Stall Causes --- "

	# create a rulebase for processing
	print "Loading Rules..."
	ruleHarness = RuleHarness.useGlobalRules("openuh/OpenUHRules.drl")

	# load the trial
	print "loading the data..."

	# check to see if the user has selected a trial
	tmp = Utilities.getCurrentTrial()
	if tmp != None:
		trial = TrialResult(tmp)
		print 
	else:
		# remove these two lines to bypass this and use the default trial
		print "No trial selected - script exiting"
		return

		# choose the right database configuration - a string which matches the end of the jdbc connection,
		# such as "perfdmf" to match "jdbc:derby:/Users/khuck/src/tau2/apple/lib/perfdmf"
		#Utilities.setSession("openuh")

		# load just the average values across all threads, input: app_name, exp_name, trial_name
		#trial = TrialResult(Utilities.getTrial("msap_parametric.optix.static", "size.400", "16.threads"))
		#trial = TrialResult(Utilities.getTrial("Fluid Dynamic - Unoptimized", "rib 45", "1_8"))

	# extract the non-callpath events from the trial
	print "extracting non-callpath..."
	extractor = ExtractNonCallpathEventOperation(trial)
	extracted = extractor.processData().get(0)

	# print "extracting event..."
	# extractor = ExtractEventOperation(trial, "MAIN__")
	# extracted = extractor.processData().get(0)

	# get basic statistics
	print "computing mean..."
	statMaker = BasicStatisticsOperation(extracted, True)
	stats = statMaker.processData()
	means = stats.get(BasicStatisticsOperation.MEAN)

	# get main event
	mainEvent = means.getMainEvent()
	print "Main Event: ", mainEvent

	# calculate all derived metrics
	print "Deriving memory stall metrics..."
	derived, PowerPerProc = getMemoryModel(means)

	# just one thread
	thread = 0

	# iterate over events, output inefficiency derived metric
	for event in derived.getEvents():
		MeanEventFact.compareEventToMain(derived, mainEvent, derived, event)
	print
	print

	# output the top 10
	top10er = TopXEvents(derived, derived.getTimeMetric(), AbstractResult.EXCLUSIVE, 10)
	top10 = top10er.processData().get(0);
	for event in top10.getEvents():
		print
		print event, "L1 hits: ", L1Hits, derived.getInclusive(0, event, L1Hits)
		print event, "L2 hits: ", L2Hits, derived.getInclusive(0, event, L2Hits)
		print event, "L3 hits: ", L3Hits, derived.getInclusive(0, event, L3Hits)
		print event, "TLB Penalty: ", TLBPenalty, derived.getInclusive(0, event, TLBPenalty)
		print event, "Local Memory Hits: ", LocalMemoryHits, derived.getInclusive(0, event, LocalMemoryHits)
		print event, "Remote Memory Hits: ", RemoteMemoryHits, derived.getInclusive(0, event, RemoteMemoryHits)
		print event, "Total: ", Total, derived.getInclusive(0, event, Total)
		print event, "Total Ratio: ", TotalRatio, derived.getInclusive(0, event, TotalRatio)
		print event, "local/remote ratio: ", RatioMemoryAccesses, derived.getInclusive(0, event, RatioMemoryAccesses)
		print

	# process the rules
	RuleHarness.getInstance().processRules()

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()