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
	Utilities.setSession("spaceghost")
	trial = Utilities.getTrial("Miranda", "BlueGeneL", "8K.old")

	# load the trial and get the metadata
	trialResult = TrialResult(trial)
	trialMetadata = TrialThreadMetadata(trial)
	return trialResult, trialMetadata

def getTop10(input):
	print "Getting top 10 events (sorted by exclusive time)..."
	getTop10 = TopXEvents(input, input.getTimeMetric(), AbstractResult.EXCLUSIVE, 10)
	top10 = getTop10.processData().get(0)
	return top10

def correlateMetadata(input, meta):
	correlator = CorrelateEventsWithMetadata(input, meta)
	outputs = correlator.processData()
	RuleHarness.getInstance().assertObject(outputs.get(0));
	return outputs

print "--------------- JPython test script start ------------"
print "doing single trial correlation analysis for Miranda on BGL"
# create a rulebase for processing
loadRules()
# load the trial
trialResult, trialMetadata = loaddata()
# extract the top 10 events, along with main, and get the event names sorted by exclusive
top10 = getTop10(trialResult)
correlateMetadata(top10, trialMetadata)
RuleHarness.getInstance().processRules()
print "---------------- JPython test script end -------------"
