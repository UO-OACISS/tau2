from glue import *
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *

True = 1
False = 0
config = "optimizer"
inApp = "optimizeme"
inExp = "test.5"
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
	metadataFields.add("A")
	metadataFields.add("B")
	#metadataFields.add("C")
	#metadataFields.add("AB")
	#metadataFields.add("AC")
	#metadataFields.add("BC")
	#metadataFields.add("ABC")
	metadataFields.add("Time")
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
r[0] = r[0]/1000000
r[1] = r[1]/1000000
r[2] = r[2]/1000000
r[3] = r[3]/1000000
#r[4] = r[4]/1000000
#r[5] = r[5]/1000000
#r[6] = r[6]/1000000
#r[7] = r[7]/1000000
#r[8] = r[8]/1000000
print r
# Using keys: [Time, AB, A, BC, C, AC, B, ABC]
# Using keys: [Time, A, C, B]
# a=3, b=1, c=1
p = 3 * r[1]
p = p + (1 * r[2])
#p = p + (1 * r[3])
p = p + r[3]
#p = p + (5 * r[5])
#p = p + (0 * r[6])
#p = p + (5 * r[7])
#p = p + r[8]
print p

print "---------------- JPython test script end -------------"
