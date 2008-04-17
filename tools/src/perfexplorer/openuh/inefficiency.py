from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TrialResult
from glue import BasicStatisticsOperation
from glue import AbstractResult
from glue import ExtractNonCallpathEventOperation
from glue import DeriveMetricOperation
from glue import MergeTrialsOperation
from glue import DerivedMetrics
from glue import TopXEvents
from glue import MeanEventFact

###################################################################

True = 1
False = 0

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

###################################################################

def getInefficiency(input):
	# set some metric names
	totalStallCycles = "BACK_END_BUBBLE_ALL"
	totalCycles = "CPU_CYCLES"
	#fpOps = "FP_OPS_RETIRED"
	fpOps = "PAPI_FP_OPS"
	instRetired = "IA64_INST_RETIRED_THIS"
	time = "LINUX_TIMERS"

	# derive the BACK_END_BUBBLE_ALL / CPU_CYCLES term
	# derive the FP_OPS_RETIRED * (BACK_END_BUBBLE_ALL / CPU_CYCLES) value
	input, tmpName = deriveMetric(input, totalStallCycles, totalCycles, DeriveMetricOperation.DIVIDE)
	input, inefficiency1 = deriveMetric(input, instRetired, tmpName, DeriveMetricOperation.MULTIPLY)
	input, inefficiency2 = deriveMetric(input, fpOps, tmpName, DeriveMetricOperation.MULTIPLY)
	#input, inefficiency3 = deriveMetric(input, totalStallCycles, time, DeriveMetricOperation.DIVIDE)

	# return the trial, and the new derived metric name
	return input, inefficiency1, inefficiency2

###################################################################

print "--------------- JPython test script start ------------"
print "--- Calculating inefficiency --- "

# create a rulebase for processing
#print "Loading Rules..."
ruleHarness = RuleHarness.useGlobalRules("openuh/OpenUHRules.drl")

# load the trial
print "loading the data..."

# choose the right database configuration - a string which matches the end of the jdbc connection,
# such as "perfdmf" to match "jdbc:derby:/Users/khuck/src/tau2/apple/lib/perfdmf"
Utilities.setSession("openuh")

# load just the average values across all threads, input: app_name, exp_name, trial_name
#trial = TrialResult(Utilities.getTrial("msap_parametric.optix.static", "size.400", "16.threads"))
trial = TrialResult(Utilities.getTrial("Fluid Dynamic - Unoptimized OpenMP", "rib 90", "Original OpenMP 1_16"))

# extract the non-callpath events from the trial
extractor = ExtractNonCallpathEventOperation(trial)
extracted = extractor.processData().get(0)

# get basic statistics
statMaker = BasicStatisticsOperation(extracted, True)
stats = statMaker.processData()
means = stats.get(BasicStatisticsOperation.MEAN)

# get main event
mainEvent = means.getMainEvent()
print "Main Event: ", mainEvent

# calculate all derived metrics
derived, inefficiency1, inefficiency2 = getInefficiency(means)

# just one thread
thread = 0

# iterate over events, output inefficiency derived metric
#print "Average", inefficiency1, "values for this trial:"
for event in derived.getEvents():
	#print event, derived.getExclusive(thread, event, inefficiency1), derived.getInclusive(thread, event, inefficiency1)
	MeanEventFact.compareEventToMain(derived, mainEvent, derived, event)
print

# process the rules
RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
