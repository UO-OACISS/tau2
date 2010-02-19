from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *

###################################################################

True = 1
False = 0

###################################################################

print "--------------- JPython test script start ------------"
print "--- Looking for load imbalances --- "

# create a rulebase for processing
print "Loading Rules..."
ruleHarness = RuleHarness.useGlobalRules("/home/khuck/tau2/tools/src/perfexplorer/openuh/BSCRules.drl")

# load the trial
print "loading the data..."

# choose the right database configuration - 
# a string which matches the end of the jdbc connection,
# such as "perfdmf" to match "jdbc:derby:/Users/khuck/src/tau2/apple/lib/perfdmf"
Utilities.setSession("local")

# load just the average values across all threads, input: app_name, exp_name, trial_name
# trial = TrialResult(Utilities.getTrial("GROMACS", "MareNostrum", "64"))
trial = DataSourceResult(Utilities.getTrial("GROMACS", "MareNostrum", "64"))

# extract the non-callpath events from the trial
trial.setIgnoreWarnings(True)
extractor = ExtractNonCallpathEventOperation(trial)
extracted = extractor.processData().get(0)

# get basic statistics
extracted.setIgnoreWarnings(True)
statMaker = BasicStatisticsOperation(extracted, False)
stats = statMaker.processData()
stddev = stats.get(BasicStatisticsOperation.STDDEV)
means = stats.get(BasicStatisticsOperation.MEAN)
totals = stats.get(BasicStatisticsOperation.TOTAL)
mainEvent = means.getMainEvent()
print "Main Event: ", mainEvent

# get the ratio between stddev and total
ratioMaker = RatioOperation(stddev, means)
ratios = ratioMaker.processData().get(0)

# iterate over events, output imbalance derived metric
thread = 0
for event in ratios.getEvents():
	for metric in ratios.getMetrics():
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
