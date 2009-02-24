from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.rules import *

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
	Utilities.setSession("localhost:5432/perfdmf")
	trial = TrialResult(Utilities.getTrial("sweep3d", "jaguar", "16"))
	#from office
	# Utilities.setSession("PERI_DB_production")
	# trial = TrialMeanResult(Utilities.getTrial("gtc", "jaguar", "64"))
	return trial

def extractNonCallpath(input):
	# extract the non-callpath events from the trial
	extractor = ExtractNonCallpathEventOperation(input)
	#extractor = ExtractCallpathEventOperation(input)
	return extractor.processData().get(0)

def getTop5(input):
	print "Getting top 5 events (sorted by exclusive time)..."
	getTop5 = TopXEvents(input, input.getTimeMetric(), AbstractResult.EXCLUSIVE, 5)
	top5 = getTop5.processData().get(0)

	return top5

def doClustering(input):
	type = AbstractResult.EXCLUSIVE
	kmeans = KMeansOperation(input, input.getTimeMetric(), type, 2)
	outputs = kmeans.processData()
	return outputs.get(0)



print "--------------- JPython test script start ------------"

print "doing single trial analysis for gtc on jaguar"

# create a rulebase for processing
# loadRules()

# load the trial
trial = loaddata()

# extract the non-callpath events
extracted = extractNonCallpath(trial)

# extract the top 5 events, along with main, and get the event names sorted by exclusive
top5 = getTop5(extracted)

clusters = doClustering(top5)

# RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
