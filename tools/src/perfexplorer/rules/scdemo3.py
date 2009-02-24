from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.rules import *
from java.util import ArrayList

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
	baseline = TrialMeanResult(Utilities.getTrial("gtc_bench", "jaguar", "64"))
	comparison = TrialMeanResult(Utilities.getTrial("gtc_bench", "jaguar", "128"))
	#from office
	# Utilities.setSession("PERI_DB_production")
	# baseline = TrialMeanResult(Utilities.getTrial("gtc", "jaguar", "64"))
	# comparison = TrialMeanResult(Utilities.getTrial("gtc", "jaguar", "128"))
	return baseline, comparison

def extractNonCallpath(input):
	# extract the non-callpath events from the trial
	extractor = ExtractNonCallpathEventOperation(input)
	#extractor = ExtractCallpathEventOperation(input)
	return extractor.processData().get(0)

def getTop5(baseline, comparison):
	print "Getting top 5 events (sorted by exclusive time)..."
	# get the top 5 events for the baseline
	getTop5 = TopXEvents(baseline, baseline.getTimeMetric(), AbstractResult.EXCLUSIVE, 5)
	baseEvents = getTop5.processData().get(0).getEvents()
	# get the top 5 events for the comparison
	getTop5 = TopXEvents(comparison, comparison.getTimeMetric(), AbstractResult.EXCLUSIVE, 5)
	compEvents = getTop5.processData().get(0).getEvents()
	# get the union of the sets
	baseEvents.addAll(compEvents)
	
	eventList = ArrayList(baseEvents)
	extractor = ExtractEventOperation(baseline, eventList)
	extractor.addInput(comparison)
	top5 = extractor.processData()

	return top5.get(0), top5.get(1)

def doScalability(baseline, comparison):
	scalability = ScalabilityOperation(baseline);
	scalability.addInput(comparison);
	scalability.reset();
	scalability.setMeasure(ScalabilityResult.Measure.SPEEDUP);
	scalability.setScaling(ScalabilityResult.Scaling.WEAK);
	return scalability.processData();

def compareMetadata(baseline, comparison):
	baseMeta = TrialMetadata(baseline.getTrial())
	compMeta = TrialMetadata(comparison.getTrial())
	oper = DifferenceMetadataOperation(baseMeta, compMeta)
	RuleHarness.getInstance().assertObject(oper)
	
def doComparison(baseline, comparison):
	diff = DifferenceOperation(baseline);
	diff.addInput(comparison);
	diff.processData();
	RuleHarness.getInstance().assertObject(diff);

print "--------------- JPython test script start ------------"

print "doing single trial analysis for gtc on jaguar"

# create a rulebase for processing
loadRules()

# load the trial
baseline, comparison = loaddata()

# compare the metadata
compareMetadata(baseline, comparison)

# extract the non-callpath events
baseline = extractNonCallpath(baseline)
comparison = extractNonCallpath(comparison)

# extract the top 5 events, along with main, and get the event names sorted by exclusive
baseline, comparison = getTop5(baseline, comparison)

# compare the trials
comp = doComparison(baseline, comparison)

# do scalability
scalability = doScalability(baseline, comparison)

RuleHarness.getInstance().assertObject(scalability.get(0));

RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
