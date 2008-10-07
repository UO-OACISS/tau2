from glue import *
from rules import *

True = 1
False = 0

def loadRules():
	print "Loading Rules..."
	ruleHarness = RuleHarness.useGlobalRules("rules/GeneralRules.drl")
	ruleHarness.addRules("rules/ApplicationRules.drl")
	ruleHarness.addRules("rules/MachineRules.drl")
	return 

def loaddata():
	print "loading the data..."
	#from home
	Utilities.setSession("localhost:5432/perfdmf")
	# from office
	Utilities.setSession("apart")
	trial = Utilities.getTrial("sweep3d", "jaguar", "256")

	# load the trial and get the metadata
	trialResult = TrialResult(trial)
	trialMetadata = TrialThreadMetadata(trial)
	return trialResult, trialMetadata

def extractCallpath(input):
	# extract the non-callpath events from the trial
	extractor = ExtractNonCallpathEventOperation(input)
	#extractor = ExtractCallpathEventOperation(input)
	return extractor.processData().get(0)

def getTop5(input):
	print "Getting top 5 events (sorted by exclusive time)..."
	getTop5 = TopXEvents(input, input.getTimeMetric(), AbstractResult.EXCLUSIVE, 5)
	top5 = getTop5.processData().get(0)

	return top5

def correlateMetadata(input, meta):
	correlator = CorrelateEventsWithMetadata(input, meta)
	outputs = correlator.processData()

	RuleHarness.getInstance().assertObject(outputs.get(0));
	return outputs



print "--------------- JPython test script start ------------"

print "doing single trial analysis for Sweep3D on jaguar"

# create a rulebase for processing
loadRules()

# load the trial
trialResult, trialMetadata = loaddata()

# extract the non-callpath events
extracted = extractCallpath(trialResult)

# extract the top 5 events, along with main, and get the event names sorted by exclusive
top5 = getTop5(extracted)

correlateMetadata(top5, trialMetadata)

RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
