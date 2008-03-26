# script for controlling perfexplorer 2.0...
# import the necessary Java classes from the glue and rules packages
from glue import PerformanceResult
from glue import Utilities
from glue import TopXEvents
from glue import TrialMeanResult
from glue import AbstractResult
from glue import DataNeeded
from glue import DeriveMetricOperation
from glue import ExtractEventOperation
from glue import ExtractNonCallpathEventOperation
from glue import ExtractCallpathEventOperation
from glue import DeriveMetricOperation
from glue import DeriveAllMetricsOperation
from glue import MergeTrialsOperation
from glue import MeanEventFact
from glue import DerivedMetrics
from rules import RuleHarness

True = 1
False = 0

def loadRules():
	global ruleHarness
	print "Loading Rules..."
	ruleHarness = RuleHarness.useGlobalRules("rules/GeneralRules.drl")
	ruleHarness.addRules("rules/ApplicationRules.drl")
	ruleHarness.addRules("rules/MachineRules.drl")
	return 

def loaddata():
	print "loading the data..."
	#from home
	# Utilities.setSession("localhost:5432/perfdmf")
	# trial = TrialMeanResult(Utilities.getTrial("gtc_bench", "jaguar", "64"))
	#from office
	Utilities.setSession("apart")
	trial = TrialMeanResult(Utilities.getTrial("gtc", "jaguar", "512"))
	return trial

def extractNonCallpath(input):
	# extract the non-callpath events from the trial
	extractor = ExtractNonCallpathEventOperation(input)
	#extractor = ExtractCallpathEventOperation(input)
	return extractor.processData().get(0)

def getTop10andMain(input):
	print "Getting top 10 events (sorted by exclusive time)..."
	getTop10 = TopXEvents(input, input.getTimeMetric(), AbstractResult.EXCLUSIVE, 10)
	top10 = getTop10.processData().get(0)
	sorted = top10.getEvents()

	# also put main in here, if it wasn't selected
	extractor = ExtractEventOperation(input, input.getMainEvent())
	justMain = extractor.processData().get(0)
	merger = MergeTrialsOperation(top10)
	merger.addInput(justMain)
	outputs = merger.processData()
	return sorted, outputs.get(0)

def deriveMetrics(input):
	derivor = DeriveAllMetricsOperation(input)
	return derivor.processData().get(0);

def processEvent(input, event, type):
	value = input.getDataPoint(0, event, input.getTimeMetric(), type)

	# get the computation density for this routine
	metric1Value = input.getDataPoint(0, event, DerivedMetrics.MFLOP_RATE, type)

	# get the L1 access rate for this routine
	metric2Value = input.getDataPoint(0, event, DerivedMetrics.MEM_ACCESSES, type)

	# get the L1 hit ratio
	metric4Value = input.getDataPoint(0, event, DerivedMetrics.L1_HIT_RATE, type)

	# get the L2 access rate for this routine
	metric5Value = input.getDataPoint(0, event, DerivedMetrics.L2_ACCESSES, type)

	# get the L2 hit ratio
	metric7Value = input.getDataPoint(0, event, DerivedMetrics.L2_HIT_RATE, type)

	# get the communication density for this routine
	# group == MPI ? value : 0.0;
	if event.startswith("MPI_"):
		metric8Value = value
	else:
		metric8Value = 0.0;
	
	print event, value, metric1Value, metric2Value, metric4Value, metric5Value, metric7Value, metric8Value

print "--------------- JPython test script start ------------"

print "doing single trial analysis for gtc on jaguar"

# create a rulebase for processing
loadRules()

# load the trial
trial = loaddata()

# extract the non-callpath events
extracted = extractNonCallpath(trial)

# extract the top 10 events, along with main, and get the event names sorted by exclusive
sorted,top10 = getTop10andMain(extracted)
top10=extracted

# calculate all derived metrics
derived = deriveMetrics(top10)

#print "EventName, value, MFlop/s, L1 Accesses/s, L1 Hit Rate, L2 Accesses/s, L2 Hit Rate, Communication"

# process main
#processEvent(derived, derived.getMainEvent(), AbstractResult.INCLUSIVE)

# process the top 10 events
for event in sorted:
	#processEvent(derived, event, AbstractResult.EXCLUSIVE)
	MeanEventFact.compareEventToMain(derived, derived.getMainEvent(), derived, event)

RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
