from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *

tauData = ""
ruleFile = "/home/khuck/tau2/tools/src/perfexplorer/openuh/BSCRules.drl"

###################################################################

def getParameters():
	global tauData
	global ruleFile
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	tmp = parameterMap.get("tauData")
	if tmp != None:
		tauData = tmp
		print "Performance data: " + tauData
	else:
		print "TAU profile data path not specified... using current directory of profile.x.x.x files."

	tmp = parameterMap.get("rules")
	if tmp != None:
		ruleFile = tmp
	else:
		print "Rule file not specified. Using default."
	print "Rules: " + ruleFile


def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

###################################################################

def main():
	global tauData
	global ruleFile
	print "--------------- JPython test script start ------------"
	print "--- Looking for load imbalances --- "

	# get the parameters
	getParameters()

	# create a rulebase for processing
	print "Loading Rules..."
	ruleHarness = RuleHarness.useGlobalRules(ruleFile)

	# load the trial
	print "loading the data..."

	# load the data
	trial = loadFile(tauData)
	trial.setIgnoreWarnings(True)

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
	maxs = stats.get(BasicStatisticsOperation.MAX)
	totals = stats.get(BasicStatisticsOperation.TOTAL)
	mainEvent = means.getMainEvent()
	print "Main Event: ", mainEvent

	# get the ratio between average and max
	ratioMaker = RatioOperation(means, maxs)
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


if __name__ == "__main__":
	main()

