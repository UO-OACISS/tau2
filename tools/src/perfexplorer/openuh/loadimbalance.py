from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TrialResult
from glue import AbstractResult
from glue import ExtractNonCallpathEventOperation
from glue import DeriveMetricOperation
from glue import MergeTrialsOperation
from glue import DerivedMetrics
from glue import MeanEventFact
from glue import BasicStatisticsOperation
from glue import RatioOperation
from glue import TopXEvents

###################################################################

True = 1
False = 0

###################################################################

print "--------------- JPython test script start ------------"
print "--- Looking for load imbalances --- "

# create a rulebase for processing
print "Loading Rules..."
ruleHarness = RuleHarness.useGlobalRules("/home/khuck/tau2/tools/src/perfexplorer/openuh/OpenUHRules.drl")

# load the trial
print "loading the data..."

# choose the right database configuration - a string which matches the end of the jdbc connection,
# such as "perfdmf" to match "jdbc:derby:/Users/khuck/src/tau2/apple/lib/perfdmf"
Utilities.setSession("openuh")

# load just the average values across all threads, input: app_name, exp_name, trial_name
trial = TrialResult(Utilities.getTrial("Fluid Dynamic", "rib 45", "1_2"))

# extract the non-callpath events from the trial
extractor = ExtractNonCallpathEventOperation(trial)
extracted = extractor.processData().get(0)

# get basic statistics
statMaker = BasicStatisticsOperation(extracted, True)
stats = statMaker.processData()
stddev = stats.get(BasicStatisticsOperation.STDDEV)
means = stats.get(BasicStatisticsOperation.MEAN)
mainEvent = means.getMainEvent()
print "Main Event: ", mainEvent

# get the ratio between stddev and total
ratioMaker = RatioOperation(stddev, means, " (STDDEV / MEAN)")
#ratioMaker = RatioOperation(stddev, means)
ratios = ratioMaker.processData().get(0)

merger = MergeTrialsOperation(means)
merger.addInput(ratios)
merged = merger.processData().get(0)

# iterate over events, output imbalance derived metric
thread = 0
metric = "CPU_CYCLES (STDDEV / MEAN)"
for event in merged.getEvents():
	#print event, merged.getExclusive(thread, event, metric)
	MeanEventFact.compareEventToMain(merged, mainEvent, merged, event, "CPU_CYCLES")
print

# process the rules
RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
