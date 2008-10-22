from glue import *
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *

True = 1
False = 0
config = "local"
inApp = "GAMESS"
# inExp = "CQoS"
inTrial = ""
parameterMap = None
fileName = "/tmp/classifier.serialized"
results = ArrayList()

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
	global results
	global inExp
	print "loading trials for experiment..."
	Utilities.setSession(config)
	trials = Utilities.getTrialsForExperiment(inApp, inExp)
	for trial in trials:
		loaded = TrialMeanResult(trial)
		results.add(loaded)
	print "...done."
	return results

def loadExperiments():
	global results
	global inExp
	print "loading experiments..."
	Utilities.setSession(config)
	experiments = Utilities.getExperimentsForApplication(inApp)
	for experiment in experiments:
		inExp = experiment.getName();
		print "processing experiment: ", inExp
		results = loadTrials()
	print "...done."
	return results

def buildClassifier(results):
	print "building classifier..."
	metadataFields = HashSet()
	metadataFields.add("molecule name")
	metadataFields.add("basis set")
	metadataFields.add("run type")
	metadataFields.add("scf type")
	metadataFields.add("node count")
	metadataFields.add("core count")
	metadataFields.add("mplevl")
	metadataFields.add("dirscf")
	# for accuracy
	# classifier = CQoSClassifierOperation(results, "accuracy", metadataFields, "basis set")
	# for performance
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.MULTILAYER_PERCEPTRON)
	classifier.processData()
	print classifier.crossValidateModel()
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.ALTERNATING_DECISION_TREE)
	classifier.processData()
	print classifier.crossValidateModel()
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.NAIVE_BAYES)
	classifier.processData()
	print classifier.crossValidateModel()
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.RANDOM_TREE)
	classifier.processData()
	print classifier.crossValidateModel()
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.SUPPORT_VECTOR_MACHINE)
	classifier.processData()
	print classifier.crossValidateModel()
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.J48)
	classifier.processData()
	print classifier.crossValidateModel()
	classifier.writeClassifier(fileName)
	print "...done."
	return classifier

def test(classifier):
	# test the classifier
	mols = ['AT', 'bz', 'bz-dimer', 'C60', 'GC', 'np', 'np-dimer']
	mps = ['MP0', 'MP2']
	for m in mols:
		for mp in mps:
			for nodes in ['1','2','4','8','16','32']:
				inputFields = HashMap()
				inputFields.put("molecule name", m)
				inputFields.put("basis set", "CCD")
				inputFields.put("run type", "ENERGY")
				inputFields.put("scf type", "RHF")
				inputFields.put("node count", nodes)
				inputFields.put("core count", "8")
				inputFields.put("mplevl", mp)
				print inputFields, "Direct / Conventional: ", classifier.getClass(inputFields),  classifier.getConfidence()


print "--------------- JPython test script start ------------"

getParameters()

print "getting trials..."

#results = loadTrials()
results = loadExperiments()

print "...done."
print "Total Trials:", results.size()

classifier = buildClassifier(results)
# test(classifier)

print "---------------- JPython test script end -------------"
