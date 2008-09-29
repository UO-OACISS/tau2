from glue import *
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *

True = 1
False = 0
config = "local"
inApp = "./ex27_2"
inExp = "0x0"
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
	metadataFields.add("cflini")
	metadataFields.add("fnorm")
	metadataFields.add("grashof")
	#metadataFields.add("ksp")
	metadataFields.add("lidvelocity")
	metadataFields.add("problemsize")
	metadataFields.add("procs")
	metadataFields.add("snes")
	# chose the linear solver
	classifier = CQoSClassifierOperation(results, "PAPI_TOT_CYC", metadataFields, "ksp")
	classifier.setClassifierType(CQoSClassifierOperation.MULTILAYER_PERCEPTRON)
	#classifier.setClassifierType(CQoSClassifierOperation.NAIVE_BAYES)
	#classifier.setClassifierType(CQoSClassifierOperation.SUPPORT_VECTOR_MACHINE)
	classifier.processData()
	classifier.writeClassifier(fileName)
	print "...done."
	return classifier


print "--------------- JPython test script start ------------"

getParameters()
results = ArrayList()

print "getting trials..."

trials = loadTrials()
for trial in trials:
	loaded = TrialMeanResult(trial)
	# important - split the trial, because it's iterative, and each iteration has its own metadata
	operator = SplitTrialPhasesOperation(loaded, "Iteration");
	outputs = operator.processData();
	for output in outputs:
		results.add(output)

# experiments = loadExperiments()
# for experiment in experiments:
	# inExp = experiment.getName();
	# print "processing experiment: ", inExp
	# trials = loadTrials()
	# for trial in trials:
		# loaded = TrialMeanResult(trial)
		# results.add(loaded)

print "...done."
print "Total Trials:", results.size()

classifier = buildClassifier(results)

# test the classifier
inputFields = HashMap()
inputFields.put("cflini", "0.1")
inputFields.put("fnorm", "0.001")
inputFields.put("grashof", "100000")
inputFields.put("lidvelocity", "100")
inputFields.put("problemsize", "0x0")
inputFields.put("procs", "2")
inputFields.put("snes", "ls")
print inputFields
print "Solver: ", classifier.getClass(inputFields),  classifier.getConfidence()


print "---------------- JPython test script end -------------"
