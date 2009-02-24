from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *
import sys
import time

True = 1
False = 0
config = "local"
inApp = "./ex27_2"
inExp = "0x0"
inTrial = ""
parameterMap = None
fileName = "/tmp/classifier.nosplit"

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

	# general properties
	# metadataFields.add("procs")
	metadataFields.add("gridsize")
	metadataFields.add("cflini")
	# metadataFields.add("prandtl")
	metadataFields.add("lidvelocity")
	metadataFields.add("grashof")
	metadataFields.add("kspmaxit")

	# non-linear properties
	metadataFields.add("snes_success")
	# metadataFields.add("snes_fnorm")  # varies wildly
	# metadataFields.add("cfl")  # varies wildly

	# linear properties
	metadataFields.add("snes")
	metadataFields.add("ksp")
	metadataFields.add("pc")
	metadataFields.add("matrixsize")
	metadataFields.add("snesrtol")
	metadataFields.add("ksprtol")
	metadataFields.add("success")
	# metadataFields.add("fnorm")  # varies wildly

	# matrix properties?
	# metadataFields.add("col-variability") # J48 .40
	# metadataFields.add("diagonal-average") # bad.
	# metadataFields.add("diagonal-sign") # bad.
	# metadataFields.add("diagonal-variance")
	# metadataFields.add("row-variability") # J48 .3974


	# chose the linear solver
	start = time.clock()
	classifier = CQoSClassifierOperation(results, "P_WALL_CLOCK_TIME", metadataFields, "ksp")
	end = time.clock()
	print end - start, " seconds to initialize classifier"
	classifier.setClassifierType(CQoSClassifierOperation.J48)
	start = time.clock()
	classifier.processData()
	end = time.clock()
	print end - start, " seconds to build classifier"
	print "validating classifier..."
	start = time.clock()
	print classifier.crossValidateModel()
	end = time.clock()
	print end - start, " seconds to validate classifier"
	classifier.writeClassifier(fileName + ".j48")
	classifier.setClassifierType(CQoSClassifierOperation.NAIVE_BAYES)
	start = time.clock()
	classifier.processData()
	end = time.clock()
	print end - start, " seconds to build classifier"
	print "validating classifier..."
	start = time.clock()
	print classifier.crossValidateModel()
	end = time.clock()
	print end - start, " seconds to validate classifier"
	classifier.writeClassifier(fileName + ".nb")
	classifier.setClassifierType(CQoSClassifierOperation.SUPPORT_VECTOR_MACHINE)
	start = time.clock()
	classifier.processData()
	end = time.clock()
	print end - start, " seconds to build classifier"
	print "validating classifier..."
	start = time.clock()
	print classifier.crossValidateModel()
	end = time.clock()
	print end - start, " seconds to validate classifier"
	classifier.writeClassifier(fileName + ".svm")
	classifier.setClassifierType(CQoSClassifierOperation.MULTILAYER_PERCEPTRON)
	start = time.clock()
	classifier.processData()
	end = time.clock()
	print end - start, " seconds to build classifier"
	print "validating classifier..."
	start = time.clock()
	print classifier.crossValidateModel()
	end = time.clock()
	print end - start, " seconds to validate classifier"
	classifier.writeClassifier(fileName + ".mp")
	print "...done."
	return classifier

def testClassifier(classifier):

	# test the classifier
	inputFields = HashMap()
	inputFields.put("snes", "ls")
	#inputFields.put("procs", "2")
	#inputFields.put("cflini", "0.1")
	#inputFields.put("prandtl", "1")
	#inputFields.put("lidvelocity", "100")
	#inputFields.put("grashof", "100000")
	#inputFields.put("fnorm", "3.266912311248e+03")
	#inputFields.put("snes_fnorm", "3562.78")
	#inputFields.put("cfl", "0.100865")
	# for lidvelocity in ['10', '50', '100']:
		# for grashof in ['100', '1000', '100000']:
			# for cflini in ['0.1', '10', '20']:
	for kspmaxit in ['200', '400', '600']:
		for gridsize in ['32x32', '64x64', '16x16']:
			for pc in ['jacobi', 'ilu', 'bjacobi', 'none', 'sor', 'asm', 'cholesky']:
				for matrixsize in ['64516x64516', '3844x3844', '15876x15876']:
					for ksprtol in ['1.000000e-04', '1.000000e-05']:
						for snesrtol in ['1.000000e-08', '1.000000e-03']:
							inputFields.put("gridsize", gridsize)
							inputFields.put("pc", pc)
							inputFields.put("matrixsize", matrixsize)
							inputFields.put("snesrtol", snesrtol)
							inputFields.put("ksprtol", ksprtol)
							# inputFields.put("lidvelocity", lidvelocity)
							# inputFields.put("grashof", grashof)
							# inputFields.put("cflini", cflini)
							inputFields.put("kspmaxit", kspmaxit)
							className =  classifier.getClass(inputFields)
							confidence = classifier.getConfidence()
							if confidence != "bcgs":
								print inputFields
								print "\tSolver: ", className,  confidence



print "--------------- JPython test script start ------------"

getParameters()
results = ArrayList()

print "getting trials..."
start = time.clock()

trials = loadTrials()
index = 1
totalTrials = trials.size()
for trial in trials:
	print "\rLoading trial ", index, "of", totalTrials,
	loaded = TrialMeanResult(trial)
	"""
	# important - split the trial, because it's iterative, and each iteration
	# has its own metadata
	operator = SplitTrialPhasesOperation(loaded, "Iteration");
	outputs = operator.processData();
	#results.add(outputs.get(outputs.size()-1))
	for output in outputs:
		results.add(output)
		break
	"""
	results.add(loaded)
	index+=1
	#if index >=12:
		#break

# experiments = loadExperiments()
# for experiment in experiments:
	# inExp = experiment.getName();
	# print "processing experiment: ", inExp
	# trials = loadTrials()
	# for trial in trials:
		# loaded = TrialMeanResult(trial)
		# results.add(loaded)

print "...done."
end = time.clock()
print end - start, " seconds to load data"
print "Total Trials:", results.size()

classifier = buildClassifier(results)
# classifier = CQoSClassifierOperation.readClassifier(fileName + ".j48")
# testClassifier(classifier)

print "---------------- JPython test script end -------------"
