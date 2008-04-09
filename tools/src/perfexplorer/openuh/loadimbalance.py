from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TrialResult
from glue import AbstractResult
from glue import ExtractNonCallpathEventOperation
from glue import ExtractCallpathEventOperation
from glue import DeriveMetricOperation
from glue import MergeTrialsOperation
from glue import DerivedMetrics
from glue import MeanEventFact
from glue import BasicStatisticsOperation
from glue import RatioOperation
from glue import TopXEvents
from rules import FactWrapper

###################################################################

True = 1
False = 0

###################################################################

print "--------------- JPython test script start ------------"
print "--- Looking for load imbalances --- "

# create a rulebase for processing
print "Loading Rules..."
ruleHarness = RuleHarness.useGlobalRules("openuh/OpenUHRules.drl")

# load the trial
print "loading the data..."

# choose the right database configuration - a string which matches the end of the jdbc connection,
# such as "perfdmf" to match "jdbc:derby:/Users/khuck/src/tau2/apple/lib/perfdmf"
Utilities.setSession("openuh")

# load just the average values across all threads, input: app_name, exp_name, trial_name
#trial = TrialResult(Utilities.getTrial("Fluid Dynamic", "rib 45", "1_8"))
trial = TrialResult(Utilities.getTrial("msap_parametric.optix.dynamic.4", "size.400", "16.threads"))
#trial = TrialResult(Utilities.getTrial("msap_parametric.optix.static_nowait", "size.400", "16.threads"))
#trial = TrialResult(Utilities.getTrial("msap_parametric.optix.dynamic.1", "size.400", "16.threads"))
#trial = TrialResult(Utilities.getCurrentTrial())

# extract the non-callpath events from the trial
extractor = ExtractNonCallpathEventOperation(trial)
extracted = extractor.processData().get(0)

# get basic statistics
statMaker = BasicStatisticsOperation(extracted, False)
stats = statMaker.processData()
stddev = stats.get(BasicStatisticsOperation.STDDEV)
means = stats.get(BasicStatisticsOperation.MEAN)
totals = stats.get(BasicStatisticsOperation.TOTAL)
mainEvent = means.getMainEvent()
print "Main Event: ", mainEvent

# get the ratio between stddev and total
ratioMaker = RatioOperation(stddev, means)
#ratioMaker = RatioOperation(stddev, means)
ratios = ratioMaker.processData().get(0)

# iterate over events, output imbalance derived metric
thread = 0
metric = "P_WALL_CLOCK_TIME"
for event in ratios.getEvents():
	#for metric in ratios.getMetrics():
	#print event, totals.getInclusive(thread, event, metric), means.getInclusive(thread, event, metric), stddev.getInclusive(thread, event, metric), ratios.getInclusive(thread, event, metric)
	MeanEventFact.evaluateLoadBalance(means, ratios, event, metric)
print

# add the callpath event names to the facts in the rulebase.

# extract the non-callpath events from the trial
extractor = ExtractCallpathEventOperation(trial)
extracted = extractor.processData().get(0)
for event in extracted.getEvents():
	fact = FactWrapper("Callpath name/value", event, None)
	RuleHarness.assertObject(fact)

# process the rules
RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
