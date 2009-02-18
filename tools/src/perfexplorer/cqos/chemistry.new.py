from glue import *
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *

True = 1
False = 0
config = "local"
inApp = "GAMESS"
inExp = "Bassi.Hiro"
inTrial = ""
parameterMap = None
fileName = "/tmp/classifier.gamess"
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
	tmp = parameterMap.get("app")
	if tmp != None:
		inApp = tmp
	tmp = parameterMap.get("exp")
	if tmp != None:
		inExp = tmp
	tmp = parameterMap.get("fileName")
	if tmp != None:
		fileName = tmp
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
	# metadataFields.add("basis set")
	# ALEX metadataFields.add("NUCLEAR REPULSION ENERGY");  # molecule dependent
	# ALEX metadataFields.add("NUMBER OF ATOMS: C");  # molecule dependent
	# ALEX metadataFields.add("NUMBER OF ATOMS: H");  # molecule dependent
	# ALEX metadataFields.add("NUMBER OF ATOMS: N");  # molecule dependent
	# ALEX metadataFields.add("NUMBER OF ATOMS: O");  # molecule dependent
	# metadataFields.add("TOTAL NUMBER OF ATOMS");  # molecule dependent
	# ALEX metadataFields.add("NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS");  # basis set dependent AND molecule dependent
	# ALEX metadataFields.add("NUMBER OF LINEARLY DEPENDENT MOS DROPPED");
	# ALEX metadataFields.add("NUMBER OF SPHERICAL CONTAMINANTS DROPPED");  # basis set dependent AND molecule dependent
	# metadataFields.add("TOTAL NUMBER OF BASIS SET SHELLS");  # basis set dependent AND molecule dependent
	# metadataFields.add("TOTAL NUMBER OF MOS IN VARIATION SPACE");  # basis set dependent AND molecule dependent
	# metadataFields.add("TOTAL NUMBER OF NONZERO TWO-ELECTRON INTEGRALS");  # basis set dependent AND molecule dependent
	# metadataFields.add("NUMBER OF ELECTRONS");  # molecule dependent
	# metadataFields.add("SPIN MULTIPLICITY");
	#metadataFields.add("run type")
	#metadataFields.add("scf type")

	metadataFields.add("NUMBER OF CARTESIAN ATOMIC ORBITALS");  # basis set dependent
	metadataFields.add("NUMBER OF OCCUPIED ORBITALS (ALPHA)");  # molecule dependent
	metadataFields.add("NUMBER OF OCCUPIED ORBITALS (BETA )");  # molecule dependent
	metadataFields.add("node count")
	metadataFields.add("core count")
	metadataFields.add("mplevl")
	metadataFields.add("dirscf")

	# for accuracy
	# classifier = CQoSClassifierOperation(results, "accuracy", metadataFields, "basis set")
	# for performance
	classifier = CQoSClassifierOperation(results, "Time", metadataFields, "dirscf")
	#classifier = CQoSClassifierOperation(results, "CPU UTILIZATION", metadataFields, "dirscf")
	#classifier = CQoSClassifierOperation(results, "CPU TIME", metadataFields, "dirscf")
	classifier.setClassifierType(CQoSClassifierOperation.ALTERNATING_DECISION_TREE)
	classifier.processData()
	classifier.writeClassifier(fileName + ".adt")
	print classifier.crossValidateModel()
	test(classifier)
	classifier.setClassifierType(CQoSClassifierOperation.NAIVE_BAYES)
	classifier.processData()
	classifier.writeClassifier(fileName + ".nb")
	print classifier.crossValidateModel()
	test(classifier)
	classifier.setClassifierType(CQoSClassifierOperation.RANDOM_TREE)
	classifier.processData()
	classifier.writeClassifier(fileName + ".rt")
	print classifier.crossValidateModel()
	test(classifier)
	classifier.setClassifierType(CQoSClassifierOperation.SUPPORT_VECTOR_MACHINE)
	classifier.processData()
	classifier.writeClassifier(fileName + ".svm")
	print classifier.crossValidateModel()
	test(classifier)
	classifier.setClassifierType(CQoSClassifierOperation.J48)
	classifier.processData()
	classifier.writeClassifier(fileName + ".j48")
	print classifier.crossValidateModel()
	test(classifier)
	classifier.setClassifierType(CQoSClassifierOperation.MULTILAYER_PERCEPTRON)
	classifier.processData()
	classifier.writeClassifier(fileName + ".mp")
	print classifier.crossValidateModel()
	test(classifier)
	classifier.writeClassifier(fileName)
	print "...done."
	return classifier

def test(classifier):
	# test the classifier
	#mols = ['bz','C60']
	mps = ['MP0', 'MP2']
	for mp in mps:
		for nodes in ['8','16','32']:
			inputFields = HashMap()
			# inputFields.put("NUCLEAR REPULSION ENERGY", "201.8371801026");
			# inputFields.put("NUMBER OF ATOMS: C", "6");
			# inputFields.put("NUMBER OF ATOMS: H", "6");
			# inputFields.put("NUMBER OF ATOMS: N", "0");
			# inputFields.put("NUMBER OF ATOMS: O", "0");
			inputFields.put("NUMBER OF CARTESIAN ATOMIC ORBITALS", "640");
			# inputFields.put("NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS", "120");
			# inputFields.put("NUMBER OF ELECTRONS", "42");
			# #inputFields.put("NUMBER OF LINEARLY DEPENDENT MOS DROPPED", "0");
			inputFields.put("NUMBER OF OCCUPIED ORBITALS (ALPHA)", "47");
			inputFields.put("NUMBER OF OCCUPIED ORBITALS (BETA )", "47");
			# #inputFields.put("NUMBER OF SPHERICAL CONTAMINANTS DROPPED", "6");
			# inputFields.put("SPIN MULTIPLICITY", "1");
			# inputFields.put("TOTAL NUMBER OF ATOMS", "12");
			# inputFields.put("TOTAL NUMBER OF BASIS SET SHELLS", "54");
			# inputFields.put("TOTAL NUMBER OF MOS IN VARIATION SPACE", "114");
			# inputFields.put("TOTAL NUMBER OF NONZERO TWO-ELECTRON INTEGRALS", "1146584");
			#inputFields.put("basis set", "CCD")
			#inputFields.put("run type", "ENERGY")
			#inputFields.put("scf type", "RHF")
			inputFields.put("node count", nodes)
			inputFields.put("core count", "8")
			inputFields.put("mplevl", mp)
			if classifier.getClass(inputFields) == "DIRECT":
				print inputFields, "Direct / Conventional: ", classifier.getClass(inputFields),  classifier.getConfidence()
	print ""

print "--------------- JPython test script start ------------"

getParameters()

print "getting trials..."

#results = loadTrials()
results = loadExperiments()

print "...done."
print "Total Trials:", results.size()

classifier = buildClassifier(results)
#test(classifier)

print "---------------- JPython test script end -------------"
