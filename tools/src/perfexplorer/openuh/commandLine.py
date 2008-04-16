from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TrialMeanResult
from glue import TopXEvents
from glue import AbstractResult
from glue import DeriveMetricOperation
from glue import DerivedMetrics
from glue import MeanEventFact

###################################################################

print "---------------- JPython test script begin -----------"

# create a rulebase for processing
#ruleHarness = RuleHarness.useGlobalRules("openuh/OpenUHRules.drl")

# load a trial
Utilities.setSession("openuh")
trial = TrialMeanResult(Utilities.getTrial("Fluid Dynamic", "rib 45", "1_8"))

# calculate the derived metric
totalStalls = "BACK_END_BUBBLE_ALL"
totalCycles = "CPU_CYCLES"
newMetric = "(BACK_END_BUBBLE_ALL/CPU_CYCLES)"
derivor = DeriveMetricOperation(trial, totalStalls, totalCycles, DeriveMetricOperation.DIVIDE)
derived = derivor.processData().get(0)

# get the top 10 events
top10er = TopXEvents(derived, newMetric, AbstractResult.EXCLUSIVE, 10)
top10 = top10er.processData().get(0)

print "Top 10 events with high stall/cycle ratios:"
for event in top10er.getSortedEventNames():
	print "\t", event, derived.getInclusive(0, event, newMetric)
	#MeanEventFact.compareEventToMain(derived, mainEvent, derived, event)

# process the rules
#RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
