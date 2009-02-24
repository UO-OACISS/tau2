from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *

True = 1
False = 0
config = "optimizer"
inApp = "optimizeme"
inExp = "test"
inTrial = ""
parameterMap = None
#fileName = "/tmp/classifier.serialized"

def getParameters():
	global parameterMap
	global config
	global inApp
	global inExp
	global fileName
	print "getting parameters..."
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	config = parameterMap.get("config")
	inApp = parameterMap.get("app")
	inExp = parameterMap.get("exp")
	fileName = parameterMap.get("fileName")
	print "...done."

def loadTrials():
	print "loading data..."
	Utilities.setSession(config)
	trials = Utilities.getTrialsForExperiment(inApp, inExp)
	print "...done."
	return trials

def loadExperiments():
	print "loading data..."
	Utilities.setSession(config)
	experiments = Utilities.getExperimentsForApplication(inApp)
	print "...done."
	return experiments

def buildClassifier(results):
	print "building classifier..."
	metadataFields = HashSet()
	metadataFields.add("Time")
	metadataFields.add("A")
	metadataFields.add("B")
	metadataFields.add("C")
	# metadataFields.add("A-B")
	# metadataFields.add("A-C")
	# metadataFields.add("B-C")
	# metadataFields.add("ABC")
	# for performance
	classifier = LinearOptimizerOperation(results, "Time", metadataFields, "Time")
	classifier.processData()
	print "...done."
	return classifier


print "--------------- JPython test script start ------------"

#getParameters()
results = ArrayList()

print "getting trials..."

trials = loadTrials()
for trial in trials:
	loaded = TrialMeanResult(trial)
	results.add(loaded)

print "...done."
print "Total Trials:", results.size()

classifier = buildClassifier(results)
r = classifier.getCoefficients()

print r
for a in range(0,11):
	for b in range(0,11):
		for c in range(0,11):
			if a+b+c == 10:
				inputFields = HashMap()
				inputFields.put("A", `a`)
				inputFields.put("B", `b`)
				inputFields.put("C", `c`)
				print a, b, c, " = ", classifier.classifyInstance(inputFields)

print "---------------- JPython test script end -------------"
