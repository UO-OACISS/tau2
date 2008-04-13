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

print "---------------- JPython test script begin -----------"

# create a rulebase for processing
ruleHarness = RuleHarness.useGlobalRules("openuh/OpenUHRules.drl")

# load a trial
trial = TrialMeanResult(Utilities.getTrial("Fluid Dynamic", "rib 45", "1_8"))

# calculate the derived metric
totalStalls = "BACK_END_BUBBLE_ALL"
totalCycles = "CPU_CYCLES"
derivor = DeriveMetricOperation(trial, totalStalls, totalCycles, DeriveMetricOperation.DIVIDE)
derived = derivor.processData().get(0)

for event in derived.getEvents():
	MeanEventFact.compareEventToMain(derived, mainEvent, derived, event)
print

# process the rules
RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
