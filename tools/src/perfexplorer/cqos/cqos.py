from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import CQoSClassifierOperation
from glue import Utilities
from glue import TrialMeanResult
from glue import AbstractResult
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0
config = "local"
inApp = "GAMESS"
inExp = "CQoS"
inTrial = ""
parameterMap = None
fileName = "/tmp/classifier.serialized"

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

def load():
	print "loading data..."
	Utilities.setSession(config)
	trials = Utilities.getTrialsForExperiment(inApp, inExp)
	print "...done."
	return trials

def buildClassifier(results):
	print "building classifier..."
	metadataFields = HashSet()
	metadataFields.add("molecule name")
	metadataFields.add("basis set")
	#metadataFields.add("run type")
	metadataFields.add("scf type")
	metadataFields.add("node count")
	metadataFields.add("core count")
	# for performance
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "scf type")
	# for accuracy
	#classifier = CQoSClassifierOperation(results, "accuracy", metadataFields, "basis set")
	#classifier.setClassifierType(CQoSClassifierOperation.NAIVE_BAYES)
	#classifier.setClassifierType(CQoSClassifierOperation.SUPPORT_VECTOR_MACHINE)
	classifier.processData()
	classifier.writeClassifier(fileName)
	print "...done."


print "--------------- JPython test script start ------------"

getParameters()
trials = load()
results = ArrayList()
print "getting trials..."
for trial in trials:
	loaded = TrialMeanResult(trial)
	results.add(loaded)
print "...done."

buildClassifier(results)

print "---------------- JPython test script end -------------"
